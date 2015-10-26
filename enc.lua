local cmd = torch.CmdLine()
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-max_seq_length', 30, 'Maximum input sentence length.')
cmd:option('-batch_size', 32, 'Training batch size.')
cmd:option('-model', 'models/enc', 'Trained autoencoder path.')
cmd:option(
  '-input', paths.concat('hdata', 'train_seg'),
  'Input data folder path.'
)
cmd:option('-embs', 'embs', 'Output embeddings path.')
cmd:option('-size', 'size', 'Output docs\' size path.')
opt = cmd:parse(arg)

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

local SPACE = 32
local NEW_LINE = 10

function load_data_into_sents(fname)
  local sents = {}
  local sizes = {}
  local size = 0

  local file = torch.DiskFile(fname, 'r')
  file:quiet()
  repeat
    local sent = file:readString('*l')
    sent = string.gsub(sent, '%s*$', '')
    if #sent ~= 0 then
      size = size + 1
      sent = stringx.split(sent)
      table.insert(sent, EOS)
      table.insert(sents, sent)
      for wid = #sent+1, params.max_seq_length do
        sents[#sents][wid] = NIL
      end
    else
      table.insert(sizes, size)
      size = 0
    end
  until file:hasError()

  return {sents, sizes}
end

local state_input
local model = {}

local function setup()
  print("Loading RNN LSTM networks...")
  local encoder = torch.load(opt.model)

  model.s_enc = {}
  model.start_s_enc = {}

  for j = 0, params.max_seq_length do
    model.s_enc[j] = {}
    for d = 1, 2 * params.layers do
      model.s_enc[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s_enc[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.encoder = encoder
  model.rnns_enc = g_cloneManyTimes(encoder, params.max_seq_length)
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil
    and model.start_s_enc ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s_enc[d]:zero()
    end
  end
end

local function _fp(state)
  g_replace_table(model.s_enc[0], model.start_s_enc)

  -- if state.pos + params.batch_size > #state.data then
  --   reset_state(state)
  -- end

  local data_batch = torch.Tensor(state.data[state.pos])
  if data_batch:size()[1] > params.max_seq_length then
    data_batch = data_batch[{{1,params.max_seq_length}}]
  end
  for data_id = 1, params.batch_size-1 do
    local new_data = state.data[state.pos + data_id]
    if new_data == nil then
      new_data = torch.zeros(params.max_seq_length)
    else
      new_data = torch.Tensor(new_data)
    end
    if new_data:size()[1] > params.max_seq_length then
      new_data = new_data[{{1,params.max_seq_length}}]
    end
    data_batch = torch.cat(data_batch, new_data, 2)
  end
  state.data_batch = transfer_data(data_batch)

  local eos_pos = transfer_data(
    torch.ones(params.batch_size):mul(params.max_seq_length)
  )
  for i = 1, params.max_seq_length do
    local x = data_batch[i]
    local s_enc = model.s_enc[i - 1]
    model.s_enc[i] = model.rnns_enc[i]:forward({x, s_enc})[2]

    -- if some sentences reach <eos> at i-th word...
    if eos_pos[x:eq(EOS)]:dim() == 1 then
      eos_pos[x:eq(EOS)] = i
    end
  end

  local emb = {}
  for d = 1, 2 * params.layers do
    emb[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  for l = 1, 2*params.layers do
    for b = 1, params.batch_size do
      emb[l][b]:copy(model.s_enc[eos_pos[b]][l][b])
    end
  end

  g_replace_table(model.start_s_enc, model.s_enc[params.max_seq_length])
  return emb
end

function mk_clean_dir(dirname)
  if paths.filep(dirname) or paths.dir(dirname) ~= nil then
    paths.rmall(dirname, 'yes')
  end
  paths.mkdir(dirname)
end

local function main()
  mk_clean_dir(opt.embs)
  mk_clean_dir(opt.size)

  g_init_gpu(opt.gpuidx)
  setup()

  print("Start encoding.")
  for filename in paths.iterfiles(opt.input) do
    print('Current file: ' .. filename)
    local data, sizes = unpack(
      load_data_into_sents(
        paths.concat(opt.input, filename)
      )
    )
    state_input = {data=data}
    reset_state(state_input)

    local len = torch.ceil(#state_input.data / params.batch_size)
    if len == 0 then
      print("  An empty file.")
    else
      local emb_out = torch.DiskFile(
        paths.concat(opt.embs, filename), "w"
      ):noAutoSpacing()
      for i = 0, len do
        local emb = _fp(state_input)

        if i ~= 0 then
          state_input.pos = state_input.pos + params.batch_size

          for l = 1, 2*params.layers do
            for b = 1, params.batch_size do
              for r = 1, params.rnn_size do
                emb_out:writeDouble(emb[l][b][r])
                emb_out:writeChar(SPACE)
              end
              emb_out:writeChar(NEW_LINE)
            end
          end
          emb_out:writeChar(NEW_LINE)
        end
      end

      local size_out = torch.DiskFile(
        paths.concat(opt.size, filename), "w"
      ):noAutoSpacing()
      for i = 1, #sizes do
        size_out:writeInt(sizes[i])
        size_out:writeChar(NEW_LINE)
      end
    end

    collectgarbage()
  end
  print("Encoding is over.")
end

main()
