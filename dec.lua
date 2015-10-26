local cmd = torch.CmdLine()
cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed.')
cmd:option('-max_seq_length', 30, 'Maximum input sentence length.')
cmd:option('-batch_size', 32, 'Training batch size.')
cmd:option('-model', 'models/dec', 'Trained autoencoder path.')
cmd:option('-input', 'embs', 'Input data (embeddings) path.')
cmd:option('-size', 'size', 'Input data (doc\'s length) path')
cmd:option('-output', 'decs', 'Output data (decoded docs) path')
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

function load_data_into_embs(fname)
  local embs = {}      --  n * 4*32*1000
  local emb = {}       --  4 * 32*1000
  local emb_layer = {} -- 32 * 1000

  local file = torch.DiskFile(fname, 'r')
  file:quiet()
  repeat
    local rnn = file:readString('*l')
    rnn = string.gsub(rnn, '%s*$', '')
    if #rnn ~= 0 then
      rnn = stringx.split(rnn)
      table.insert(emb_layer, rnn)
      if #emb_layer == params.batch_size then
        table.insert(emb, emb_layer)
        emb_layer = {}
      end
    elseif #emb ~= 0 then
      table.insert(embs, emb)
      emb = {}
    end
  until file:hasError()

  return embs
end

local state_input
local model = {}

local function setup()
  print("Loading RNN LSTM networks...")
  local decoder = torch.load(opt.model)

  model.s_dec = {}

  for j = 0, params.max_seq_length do
    model.s_dec[j] = {}
    for d = 1, 2 * params.layers do
      model.s_dec[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end

  model.decoder = decoder
  model.rnns_dec = g_cloneManyTimes(decoder, params.max_seq_length)
end

local function reset_state(state)
  state.pos = 1
end

local function _fp(state)
  local words = nil
  local embs = {}
  for l = 1, 2*params.layers do
    table.insert(
      embs, transfer_data(torch.Tensor(state.data[state.pos][l]))
    )
  end

  g_replace_table(model.s_dec[0], embs)

  for i = 1, params.max_seq_length do
    local y = transfer_data(torch.zeros(params.batch_size))
    local s_dec = model.s_dec[i - 1]
    local emb = s_dec[2*params.layers]
    local pred = nil -- 32*25002
    _, model.s_dec[i], pred = unpack(model.rnns_dec[i]:forward(
      {emb, y, s_dec}
    ))
    local _, word = pred:max(2) -- 32*1
    if words == nil then
      words = word:double()
    else
      words = torch.cat(words, word:double())
    end
  end
  return words
end

function mk_clean_dir(dirname)
  if paths.filep(dirname) or paths.dir(dirname) ~= nil then
    paths.rmall(dirname, 'yes')
  end
  paths.mkdir(dirname)
end

local function main()
  mk_clean_dir(opt.output)

  g_init_gpu(opt.gpuidx)
  setup()

  print("Starting decoding.")
  for filename in paths.iterfiles(opt.input) do
    print('Current file: ' .. filename)
    state_input = {data=load_data_into_embs(
      paths.concat(opt.input, filename)
    )}

    local sizes = {}
    local size_in = torch.DiskFile(
      paths.concat(opt.size, filename), "r"
    )
    size_in:quiet()
    repeat
      local size = size_in:readString('*l')
      size = string.gsub(size, '%s*$', '')
      size = tonumber(size)
      table.insert(sizes, size)
    until size_in:hasError()

    reset_state(state_input)

    local line_printed = 0
    local doc_id = 1
    local line_newline = sizes[doc_id]
    local dec_out = torch.DiskFile(
      paths.concat(opt.output, filename), "w"
    ):noAutoSpacing()
    for i = 0, #state_input.data do
      local words = _fp(state_input)
      if i ~= 0 then
        state_input.pos = state_input.pos + 1
        for b = 1, params.batch_size do
          for w = 1, params.max_seq_length do
            dec_out:writeInt(words[b][w])
            dec_out:writeChar(SPACE)
          end
          dec_out:writeChar(NEW_LINE)
          line_printed = line_printed + 1
          if line_printed == line_newline then
            dec_out:writeChar(NEW_LINE)
            if doc_id < #sizes then
              doc_id = doc_id + 1
              line_newline = line_newline + sizes[doc_id]
            else
              break
            end
          end
        end
      end
    end

    collectgarbage()
  end

  print("Decoding is over.")
end

main()
