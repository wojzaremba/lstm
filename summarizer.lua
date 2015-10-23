local cmd = torch.CmdLine()
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-max_seq_length', 30, 'Maximum input sentence length.')
cmd:option('-batch_size', 32, 'Training batch size.')
cmd:option('-model_path', 'models', 'Trained autoencoder path.')
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
      table.insert(sent, '<eos>')
      table.insert(doc, sent)
    else
      table.insert(doc[#doc], '<eod>')
      table.insert(docs, doc)
      doc = {}
    end
  until file:hasError()

  return docs
end

local function load_data_into_sents(fname)
  local sents = {}

  local file = torch.DiskFile(fname, 'r')
  file:quiet()
  repeat
    local sent = file:readString('*l')
    sent = string.gsub(sent, '%s*$', '')
    if #sent ~= 0 then
      sent = stringx.split(sent)
      if #sent < params.max_seq_length then
        table.insert(sent, EOS)
        table.insert(sents, sent)
      end
      for wid = #sent+1, params.max_seq_length do
        sents[#sents][wid] = NIL
      end
    end
  until file:hasError()

  return sents
end

local state_train, state_valid, state_test
local model = {}
local paramx_enc, paramdx_enc, paramx_dec, paramdx_dec

local function setup()
  print("Loading RNN LSTM networks...")
  local encoder = torch.load(path.concat(opt.model_path, 'enc.net'))
  local decoder = torch.load(path.concat(opt.model_path, 'dec.net'))

  paramx_enc, paramdx_enc = encoder:getParameters()
  paramx_dec, paramdx_dec = decoder:getParameters()

  model.s_enc = {}
  model.ds_enc = {}
  model.start_s_enc = {}
  model.s_dec = {}
  model.ds_dec = {}
  model.start_s_dec = {}
  model.embeddings = {}

  for j = 0, params.max_seq_length do
    model.s_enc[j] = {}
    model.s_dec[j] = {}
    for d = 1, 2 * params.layers do
      model.s_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
      model.s_dec[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end

    model.embeddings[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  for d = 1, 2 * params.layers do
    model.start_s_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.start_s_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.encoder = encoder
  model.decoder = decoder
  model.rnns_enc = g_cloneManyTimes(encoder, params.max_seq_length)
  model.rnns_dec = g_cloneManyTimes(decoder, params.max_seq_length)
  model.norm_dw_enc = 0
  model.norm_dw_dec = 0
  model.err = transfer_data(torch.zeros(params.max_seq_length))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil
    and model.start_s_enc ~= nil
    and model.start_s_dec ~= nil then
    for d = 1, 2 * params.layers do
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

local function _fp(state)
  g_replace_table(model.s_enc[0], model.start_s_enc)

  if state.pos + params.batch_size > #state.data then
    reset_state(state)
  end

  local data_batch = torch.Tensor(state.data[state.pos])
  for data_id = 1, params.batch_size-1 do
    data_batch = torch.cat(
      data_batch, torch.Tensor(state.data[state.pos + data_id]), 2
    )
  end
  state.data_batch = transfer_data(data_batch)

  local eos_pos = transfer_data(
    torch.ones(params.batch_size):mul(params.max_seq_length)
  )
  for i = 1, params.max_seq_length do
    local x = state.data_batch[i]
    local s_enc = model.s_enc[i - 1]
    model.embeddings[i], model.s_enc[i] = unpack(
      model.rnns_enc[i]:forward({x, s_enc})
    )

    -- if some sentences reach <eos> at i-th word...
    if eos_pos[x:eq(EOS)]:dim() == 1 then
      eos_pos[x:eq(EOS)] = i
    end
  end

  for l = 1, 2*params.layers do
    for b = 1, params.batch_size do
      model.s_dec[0][l][b]:copy(model.s_enc[eos_pos[b]][l][b])
    end
  end

  for i = 1, params.max_seq_length do
    local y = state.data_batch[i]
    local s_dec = model.s_dec[i - 1]
    local emb = s_dec[2*params.layers]
    model.err[i], model.s_dec[i] = unpack(
      model.rnns_dec[i]:forward({emb, y, s_dec})
    )
  end
  g_replace_table(model.start_s_enc, model.s_enc[params.max_seq_length])

  return model.err:mean()
end

local function _bp(state)
  paramdx_enc:zero()
  paramdx_dec:zero()
  reset_ds()

  for i = params.max_seq_length, 1, -1 do
    local y = state.data_batch[i]
    local s_dec = model.s_dec[i - 1]
    local embeddings = s_dec[2*params.layers]
    local d_pred = transfer_data(
      torch.zeros(params.batch_size, params.vocab_size)
    )
    local derr = transfer_data(torch.ones(1))

    local tmp_dec
    tmp_dec = model.rnns_dec[i]:backward(
      {embeddings, y, s_dec},
      {derr, model.ds_dec, d_pred}
    )[3]

    g_replace_table(model.ds_dec, tmp_dec)
    cutorch.synchronize()
  end

  g_replace_table(model.ds_enc, model.ds_dec)

  for i = params.max_seq_length, 1, -1 do
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
  local perp = 0
  for i = 1, len do
    perp = perp + _fp(state_valid)
    state_valid.pos = state_valid.pos + params.batch_size
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
  local len = #state_test.data / params.batch_size
  local perp = 0
  for i = 1, len do
    perp = perp + _fp(state_test)
    state_test.pos = state_test.pos + params.batch_size
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
end

local function main()
  g_init_gpu(opt.gpuidx)

  local filenames = {}
  for filename in paths.iterfiles(opt.train) do
    table.insert(filenames, filename)
  end
  local num_of_sents = 0
  for id, f in pairs(filenames) do
    local sents = load_data_into_sents(paths.concat(opt.train, f))
    num_of_sents = num_of_sents + #sents
    collectgarbage()
  end
  collectgarbage()
  local epoch_size = torch.floor(num_of_sents / params.batch_size)

  local train_sents = load_data_into_sents(
    paths.concat(opt.train, filenames[1])
  )
  local valid_sents = load_data_into_sents(opt.valid)
  local test_sents = load_data_into_sents(opt.test)
  state_train = {data=train_sents}
  state_valid = {data=valid_sents}
  state_test = {data=test_sents}
  setup()
  reset_state(state_train)

  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")

  local perps
  local file_step = 1
  local file_size = torch.floor(#state_train.data / params.batch_size)
  while epoch < params.max_max_epoch do
    local perp = _fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    _bp(state_train)

    total_cases = total_cases + params.max_seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
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

      print('Current file: ' .. filenames[file_step] ..
            ', file no. ' .. file_step ..
            ', current step: ' .. step)

      state_train.data = load_data_into_sents(
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
