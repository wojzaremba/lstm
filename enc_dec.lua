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
        table.insert(sent, params.vocab_size-1)
        table.insert(sents, sent)
      end
      for wid = #sent+1, params.max_seq_length do
        sents[#sents][wid] = params.vocab_size
      end
    end
  until file:hasError()

  return sents
end

local state_test
local model = {}
local paramx_enc, paramdx_enc, paramx_dec, paramdx_dec
local embeddings = {}
local preds = {}

local function setup()
  print("Loading RNN LSTM networks...")
  local encoder = torch.load('./models/12.enc')
  local decoder = torch.load('./models/12.dec')

  paramx_enc, paramdx_enc = encoder:getParameters()
  paramx_dec, paramdx_dec = decoder:getParameters()

  model.s_enc = {}
  model.ds_enc = {}
  model.start_s_enc = {}
  model.s_dec = {}
  model.ds_dec = {}
  model.start_s_dec = {}
  model.embeddings = {}
  model.preds = {}

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
  g_replace_table(model.s_dec[0], model.start_s_dec)

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

  for i = 1, params.max_seq_length do
    local x = state.data_batch[i]
    local y = state.data_batch[i]
    local s_enc = model.s_enc[i - 1]
    local s_dec = model.s_dec[i - 1]
    model.embeddings[i], model.s_enc[i] = unpack(
      model.rnns_enc[i]:forward({x, s_enc})
    )
    model.err[i], model.s_dec[i], model.preds[i] = unpack(
      model.rnns_dec[i]:forward({model.embeddings[i], y, s_dec})
    )
  end
  g_replace_table(model.start_s_enc, model.s_enc[params.max_seq_length])
  g_replace_table(model.start_s_dec, model.s_dec[params.max_seq_length])

  return model.err:mean()
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns_enc)
  g_disable_dropout(model.rnns_dec)
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

  local fout = torch.DiskFile("preds", "w"):noAutoSpacing()
  for preds_size = 1, #preds do
    for b = 1, params.batch_size do
      for s = 1, params.max_seq_length do
        fout:writeInt(preds[preds_size][b][s])
        fout:writeChar(32)
      end
      fout:writeChar(10)
    end
    fout:writeChar(10)
  end
end

main()
