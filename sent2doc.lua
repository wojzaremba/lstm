local cmd = torch.CmdLine()
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-max_seq_length', 30, 'Maximum input sentence length.')
cmd:option('-max_doc_length', 20, 'Maximum input document length.')
cmd:option('-batch_size', 32, 'Data batch size.')
cmd:option(
  '-train', paths.concat('hdata', 'train_seg'),
  'Training data folder path.'
)
cmd:option(
  '-valid', paths.concat('hdata', "valid_permute_segment.txt"),
  'Validation data path.'
)
cmd:option(
  '-test', paths.concat('hdata', "test_permute_segment.txt"),
  'Testing data path.'
)
cmd:option('-encoder', 'models/sent_enc.net', 'Sentence encoder path.')
local opt = cmd:parse(arg)

local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn")
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('paths')
require('nngraph')
require('base')

local function transfer_data(x)
  return x:cuda()
end

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                max_seq_length=tonumber(arg[2]),
                layers=2,
                decay=1.15,
                rnn_size=1000,
                dropout=0.65,
                init_weight=0.08,
                lr=0.1,
                vocab_size=25002,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}

--]]
-- Trains 1h and gives test 115 perplexity.
-- [[
local params = {batch_size=tonumber(opt.batch_size),
                max_seq_length=tonumber(opt.max_seq_length),
                max_doc_length=tonumber(opt.max_doc_length),
                layers=2,
                decay=2,
                rnn_size=1000,
                dropout=0.2,
                init_weight=0.08,
                lr=0.1,
                vocab_size=25002,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

local word_emb_size = 2*params.layers*params.rnn_size
local stringx = require('pl.stringx')
local EOS = params.vocab_size-1
local NIL = params.vocab_size

local function load_data_into_docs(fname)
  local docs = {}
  local doc = {}

  local file = torch.DiskFile(fname, 'r')
  file:quiet()
  repeat
    local sent = file:readString('*l')
    sent = string.gsub(sent, '%s*$', '')
    if #sent ~= 0 then
      sent = stringx.split(sent)
      table.insert(sent, EOS)
      for wid = #sent+1, params.max_seq_length do
        table.insert(sent, NIL)
      end
      table.insert(doc, sent)
    else
      table.insert(docs, doc)
      doc = {}
    end
  until file:hasError()

  return docs
end

local state_train, state_valid, state_test
local model = {}
local paramx_enc, paramdx_enc, paramx_dec, paramdx_dec

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s_enc       = nn.Identity()()
  local i_0              = nn.Linear(word_emb_size, params.rnn_size)(x)
  -- local i                = {[0] = nn.Sigmoid()(i_0)}
  local i                = {[0] = i_0}
  local next_s_enc       = {}
  local split_enc        = {prev_s_enc:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split_enc[2 * layer_idx - 1]
    local prev_h         = split_enc[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s_enc, next_c)
    table.insert(next_s_enc, next_h)
    i[layer_idx] = next_h
  end

  local encoder = nn.gModule(
    {x, prev_s_enc},
    {i[params.layers], nn.Identity()(next_s_enc)}
  )

  local prev_s_dec      = nn.Identity()()
  local j_0             = nn.Identity()()
  local j               = {[0] = j_0}
  local next_s_dec      = {}
  local split_dec       = {prev_s_dec:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split_dec[2 * layer_idx - 1]
    local prev_h         = split_dec[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(j[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s_dec, next_c)
    table.insert(next_s_dec, next_h)
    j[layer_idx] = next_h
  end

  local h2y              = nn.Linear(params.rnn_size, word_emb_size)
  local dropped          = nn.Dropout(params.dropout)(j[params.layers])
  local pred             = h2y(dropped)
  -- local pred             = nn.LogSoftMax()(h2y(dropped))
  -- local mask             = torch.ones(params.vocab_size)
  -- mask[NIL] = 0
  -- local err              = nn.ClassNLLCriterion(mask)({pred, y})
  local err              = nn.MSECriterion()({pred, y})

  local decoder = nn.gModule(
    {j_0, y, prev_s_dec},
    {err, nn.Identity()(next_s_dec), pred}
  )

  encoder:getParameters():uniform(-params.init_weight, params.init_weight)
  decoder:getParameters():uniform(-params.init_weight, params.init_weight)
  encoder = transfer_data(encoder)
  decoder = transfer_data(decoder)

  return {encoder, decoder}
end

local function setup()
  print("Creating a RNN LSTM network.")
  local sent_encoder = torch.load(opt.encoder)
  local encoder, decoder = unpack(create_network())
  paramx_enc, paramdx_enc = encoder:getParameters()
  paramx_dec, paramdx_dec = decoder:getParameters()

  model.s_sent_enc = {}
  model.start_s_sent_enc = {}

  for j = 0, params.max_seq_length do
    model.s_sent_enc[j] = {}
    for d = 1, 2 * params.layers do
      model.s_sent_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s_sent_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.s_enc = {}
  model.ds_enc = {}
  model.start_s_enc = {}
  model.s_dec = {}
  model.ds_dec = {}
  model.start_s_dec = {}

  for j = 0, params.max_doc_length do
    model.s_enc[j] = {}
    model.s_dec[j] = {}
    for d = 1, 2 * params.layers do
      model.s_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
      model.s_dec[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.start_s_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.sent_encoder = sent_encoder
  model.encoder = encoder
  model.decoder = decoder
  model.rnns_sent_enc = g_cloneManyTimes(sent_encoder, params.max_seq_length)
  model.rnns_enc = g_cloneManyTimes(encoder, params.max_doc_length)
  model.rnns_dec = g_cloneManyTimes(decoder, params.max_doc_length)
  model.norm_dw_enc = 0
  model.norm_dw_dec = 0
  model.err = transfer_data(torch.zeros(params.max_doc_length))
end

local function reset_state(state)
  state.pos = 0
  if model ~= nil
    and model.start_s_sent_enc ~= nil
    and model.start_s_enc ~= nil
    and model.start_s_dec ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s_sent_enc[d]:zero()
      model.start_s_enc[d]:zero()
      model.start_s_dec[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds_enc do
    model.ds_enc[d]:zero()
    model.ds_dec[d]:zero()
  end
end

local function get_embedding_batch(state)
  g_replace_table(model.s_sent_enc[0], model.start_s_sent_enc)
  local doc_batch = {}
  local eod_pos = torch.ones(params.batch_size):mul(params.max_doc_length)

  for b = 1, params.batch_size do
    if #state.data[state.pos+b] < params.max_doc_length then
      eod_pos[b] = #state.data[state.pos+b]+1
    end
  end

  for s = 1, params.max_doc_length do
    local sent_batch
    for b = 1, params.batch_size do
      local new_data = state.data[state.pos+b][s]
      if new_data ~= nil then
        new_data = torch.Tensor(new_data)
      else
        new_data = torch.zeros(params.max_seq_length)
      end

      if new_data:size()[1] > params.max_seq_length then
        new_data = new_data[{{1,params.max_seq_length}}]
      end

      if sent_batch ~= nil then
        sent_batch = torch.cat(sent_batch, new_data, 2)
      else
        sent_batch = new_data
      end
    end

    sent_batch = transfer_data(sent_batch)
    local eos_pos = transfer_data(
      torch.ones(params.batch_size):mul(params.max_seq_length)
    )

    sent_batch = transfer_data(sent_batch)
    for i = 1, params.max_seq_length do
      local x = sent_batch[i]
      local s_sent_enc = model.s_sent_enc[i - 1]
      _, model.s_sent_enc[i] = unpack(
        model.rnns_sent_enc[i]:forward({x, s_sent_enc})
      )

      -- if some sentences reach <eos> at i-th word...
      if eos_pos[x:eq(EOS)]:dim() == 1 then
        eos_pos[x:eq(EOS)] = i
      end
    end

    local emb = transfer_data(torch.zeros(params.batch_size, word_emb_size))

    for l = 1, 2*params.layers do
      for b = 1, params.batch_size do
        local start_id = (l-1)*params.rnn_size+1
        local end_id   =  l   *params.rnn_size
        emb[b][{{start_id, end_id}}]:copy(model.s_sent_enc[eos_pos[b]][l][b])
        -- model.start_s_sent_enc[b]:copy()
      end
    end

    table.insert(doc_batch, emb)
    g_replace_table(
      model.start_s_sent_enc, model.s_sent_enc[params.max_seq_length]
    )
  end

  return {doc_batch, eod_pos}
end

local function _fp(state)
  g_replace_table(model.s_enc[0], model.start_s_enc)

  if state.pos + params.batch_size > #state.data then
    reset_state(state)
  end

  -- doc_len * batch_size * 4000
  local data_batch, eod_pos = unpack(get_embedding_batch(state))

  -- local data_batch = torch.Tensor(state.data[state.pos])
  -- for data_id = 1, params.batch_size-1 do
  --   data_batch = torch.cat(
  --     data_batch, torch.Tensor(state.data[state.pos + data_id]), 2
  --   )
  -- end
  state.data_batch = data_batch

  -- local eos_pos = transfer_data(
  --   torch.ones(params.doc_batch_size):mul(params.max_doc_length)
  -- )
  for i = 1, params.max_doc_length do
    local x = state.data_batch[i]
    local s_enc = model.s_enc[i - 1]
    _, model.s_enc[i] = unpack(
      model.rnns_enc[i]:forward({x, s_enc})
    )

    -- if some sentences reach <eos> at i-th word...
    -- if eos_pos[x:eq(EOS)]:dim() == 1 then
    --   eos_pos[x:eq(EOS)] = i
    -- end
  end

  for l = 1, 2*params.layers do
    for b = 1, params.batch_size do
      model.s_dec[0][l][b]:copy(model.s_enc[eod_pos[b]][l][b])
      model.start_s_enc[l][b]:copy(model.s_enc[eod_pos[b]][l][b])
    end
  end

  for i = 1, params.max_doc_length do
    local y = state.data_batch[i]
    local s_dec = model.s_dec[i - 1]
    local emb = s_dec[2*params.layers]
    model.err[i], model.s_dec[i] = unpack(
      model.rnns_dec[i]:forward({emb, y, s_dec})
    )
  end
  -- g_replace_table(model.start_s_enc, model.s_enc[params.max_doc_length])

  return model.err:mean()
end

local function _bp(state)
  paramdx_enc:zero()
  paramdx_dec:zero()
  reset_ds()

  for i = params.max_doc_length, 1, -1 do
    local y = state.data_batch[i]
    local s_dec = model.s_dec[i - 1]
    local embeddings = s_dec[2*params.layers]
    local d_pred = transfer_data(
      torch.zeros(params.batch_size, word_emb_size)
    )
    local derr = transfer_data(torch.ones(1))

    local tmp_dec = model.rnns_dec[i]:backward(
      {embeddings, y, s_dec},
      {derr, model.ds_dec, d_pred}
    )[3]

    g_replace_table(model.ds_dec, tmp_dec)
    cutorch.synchronize()
  end

  g_replace_table(model.ds_enc, model.ds_dec)

  for i = params.max_doc_length, 1, -1 do
    local x = state.data_batch[i]
    local s_enc = model.s_enc[i - 1]
    local d_embedding = transfer_data(
      torch.zeros(params.batch_size, params.rnn_size)
    )
    local tmp_enc = model.rnns_enc[i]:backward(
      {x, s_enc},
      {d_embedding, model.ds_enc}
    )[2]

    g_replace_table(model.ds_enc, tmp_enc)
    cutorch.synchronize()
  end

  state.pos = state.pos + params.batch_size
  model.norm_dw_enc = paramdx_enc:norm()
  model.norm_dw_dec = paramdx_dec:norm()
  if model.norm_dw_enc > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw_enc
    paramdx_enc:mul(shrink_factor)
  end
  if model.norm_dw_dec > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw_dec
    paramdx_dec:mul(shrink_factor)
  end
  paramx_enc:add(paramdx_enc:mul(-params.lr))
  paramx_dec:add(paramdx_dec:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
  local len = #state_valid.data / params.batch_size
  local err = 0
  for i = 1, len do
    err = err + _fp(state_valid)
    state_valid.pos = state_valid.pos + params.batch_size
  end
  print("Validation set error : " .. g_f3(torch.exp(err / len)))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
  local len = #state_test.data / params.batch_size
  local err = 0
  for i = 1, len do
    err = err + _fp(state_test)
    state_test.pos = state_test.pos + params.batch_size
  end
  print("Test set error : " .. g_f3(torch.exp(err / len)))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
end

local function main()
  g_init_gpu(opt.gpuidx)

  local filenames = {}
  for filename in paths.iterfiles(opt.train) do
    table.insert(filenames, filename)
  end
  local num_of_docs = 0
  for id, f in pairs(filenames) do
    local docs = load_data_into_docs(paths.concat(opt.train, f))
    num_of_docs = num_of_docs + #docs
    collectgarbage()
  end
  collectgarbage()
  local epoch_size = torch.floor(num_of_docs / params.batch_size)

  local train_docs = load_data_into_docs(
    paths.concat(opt.train, filenames[1])
  )
  local valid_docs = load_data_into_docs(opt.valid)
  local test_docs = load_data_into_docs(opt.test)
  state_train = {data=train_docs}
  state_valid = {data=valid_docs}
  state_test = {data=test_docs}
  setup()
  reset_state(state_train)

  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")

  local errs
  local file_step = 1
  local file_size = torch.floor(#state_train.data / params.batch_size)
  while epoch < params.max_max_epoch do
    local err = _fp(state_train)
    if errs == nil then
      errs = torch.zeros(epoch_size):add(err)
    end
    errs[step % epoch_size + 1] = err
    step = step + 1
    _bp(state_train)

    total_cases = total_cases + params.max_doc_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train err. = ' .. g_f3(torch.exp(errs:mean())) ..
            ', wps = ' .. wps ..
            ', encoder dw:norm() = ' .. g_f3(model.norm_dw_enc) ..
            ', decoder dw:norm() = ' .. g_f3(model.norm_dw_dec) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end

      torch.save(
        'models/'..tostring(torch.floor(epoch))..'.enc', model.encoder
      )
      torch.save(
        'models/'..tostring(torch.floor(epoch))..'.dec', model.decoder
      )
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end

    if step == file_size then
      file_step = file_step + 1
      if file_step > #filenames then
        file_step = 1
      end

      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('Current file: ' .. filenames[file_step] ..
            ', file no. ' .. file_step ..
            ', current step: ' .. step ..
            ', since beginning = ' .. since_beginning .. ' mins.')

      state_train.data = load_data_into_docs(
        paths.concat(opt.train, filenames[file_step])
      )
      state_train.pos = 1
      file_size = file_size + torch.floor(
        #state_train.data / params.batch_size
      )
    end
  end
  run_test()
  print("Training is over.")
end

main()
