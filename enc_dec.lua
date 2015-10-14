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
local params = {batch_size=32,
                max_seq_length=tonumber(arg[2]),
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
local data_path = "./hdata/"
local EOS = params.vocab_size-1
local NIL = params.vocab_size

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

local state_test
local model = {}
local paramx_enc, paramdx_enc, paramx_dec, paramdx_dec, paramx_dec_enc, paramdx_dec_enc
local embeddings = {}
local preds = {}
local ys = {}

local function setup()
  print("Loading RNN LSTM networks...")
  local encoder = torch.load('./models/13.enc')
  local decoder = torch.load('./models/13.dec')
  local decoder_enc = torch.load('./models/13.enc')

  paramx_enc, paramdx_enc = encoder:getParameters()
  paramx_dec, paramdx_dec = decoder:getParameters()
  paramx_dec_enc, paramdx_dec_enc = decoder_enc:getParameters() 

  model.s_enc = {}
  model.start_s_enc = {}
  model.s_dec = {}
  model.start_s_dec = {}
  model.s_dec_enc = {}
  model.start_s_dec_enc = {}
  model.embeddings = {}
  model.preds = {}

  for j = 0, params.max_seq_length do
    model.s_enc[j] = {}
    model.s_dec[j] = {}
    model.s_dec_enc[j] = {}
    for d = 1, 2 * params.layers do
      model.s_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
      model.s_dec[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
      model.s_dec_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end

    model.embeddings[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  for d = 1, 2 * params.layers do
    model.start_s_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.start_s_dec[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.start_s_dec_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.encoder = encoder
  model.decoder = decoder
  model.decoder_enc = decoder_enc
  model.rnns_enc = g_cloneManyTimes(encoder, params.max_seq_length)
  model.rnns_dec = g_cloneManyTimes(decoder, params.max_seq_length)
  model.rnns_dec_enc = g_cloneManyTimes(decoder_enc, params.max_seq_length)
  model.err = transfer_data(torch.zeros(params.max_seq_length))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil
    and model.start_s_enc ~= nil
    and model.start_s_dec ~= nil 
    and model.start_s_dec_enc ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s_enc[d]:zero()
      model.start_s_dec[d]:zero()
      model.start_s_dec_enc[d]:zero()
    end
  end
end

local function _fp(state)
  g_replace_table(model.s_enc[0], model.start_s_enc)
  g_replace_table(model.s_dec_enc[0], model.start_s_dec_enc)

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
  local null = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  model.err[1], model.s_dec[1], model.preds[1] = unpack(
    model.rnns_dec[1]:forward(
      {null, state.data_batch[1], model.s_dec[0]}
    )
  )
  _, model.s_dec_enc[1] = unpack(model.rnns_dec_enc[1]:forward({state.data_batch[1], model.s_dec_enc[0]}))

  for i = 2, params.max_seq_length do
    local _, predicted_word = model.preds[i-1]:max(2)
    local predicted_embedding
    predicted_embedding, model.s_dec_enc[i] = unpack(
      model.rnns_dec_enc[i]:forward({predicted_word:select(2, 1), model.s_dec_enc[i-1]})
    )

    local y = state.data_batch[i]
    local s_dec = model.s_dec[i - 1]
    model.err[i], model.s_dec[i], model.preds[i] = unpack(
      model.rnns_dec[i]:forward({predicted_embedding, y, s_dec})
    )
  end

  table.insert(ys, state.data_batch)

  g_replace_table(model.start_s_enc, model.s_enc[params.max_seq_length])
  g_replace_table(model.start_s_dec_enc, model.s_dec_enc[params.max_seq_length])

  return model.err:mean()
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
  g_disable_dropout(model.rnns_dec_enc)
  local len = #state_test.data / params.batch_size
  local perp = 0
  local ans

  for i = 0, len do
    ans = nil
    local p = _fp(state_test)

    if i ~= 0 then
      state_test.pos = state_test.pos + params.batch_size
      perp = perp + p

      for j = 1, params.max_seq_length do
        local val, id = model.preds[j]:max(2)
        if ans == nil then
          ans = id:double()
        else
          ans = torch.cat(ans, id:double())
        end
      end
      table.insert(preds, ans)
    end
  end

  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len-1) )))
  g_enable_dropout(model.rnns_enc)
  g_enable_dropout(model.rnns_dec)
  g_enable_dropout(model.rnns_dec_enc)
end

local function main()
  g_init_gpu(arg)

  local test_sents = load_data_into_sents(
    data_path .. "test_permute_segment.txt"
  )
  state_test = {data=test_sents}
  setup()
  print("Start testing.")
  run_test()
  print("Testing is over.")

  local pred_out = torch.DiskFile("preds", "w"):noAutoSpacing()
  local ref_out = torch.DiskFile("ref", "w"):noAutoSpacing()
  for preds_size = 1, #preds do
    for b = 1, params.batch_size do
      for s = 1, params.max_seq_length do
        pred_out:writeInt(preds[preds_size][b][s])
        pred_out:writeChar(32)
        ref_out:writeInt(ys[preds_size+1][s][b])
        ref_out:writeChar(32)
      end
      pred_out:writeChar(10)
      ref_out:writeChar(10)
    end
    pred_out:writeChar(10)
    ref_out:writeChar(10)
  end
end

main()
