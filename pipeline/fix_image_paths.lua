-- Pandoc Lua filter: normalise image paths, widths, code-block UTF-8, and
-- prefix heading IDs with the chapter slug to avoid `\label{introduction}`
-- collisions across chapters. The prefix is taken from the THESIS_CH_PREFIX
-- environment variable set by the Makefile per pandoc invocation.
--
-- The chapter markdown references images at "../figures/<name>.ext". The
-- master LaTeX declares `\graphicspath{{figures/}}`, so the prefix must be
-- stripped: pandoc would otherwise emit `\includegraphics{../figures/x.png}`
-- which won't resolve from the Overleaf project's root.
--
-- Also: if the image has no explicit width attribute, default to
-- `width=\linewidth`. Authors who need a different width can set
-- `{ width=0.7\linewidth }` in markdown and we leave it alone.

local function strip_prefix(src)
  -- Remove "../figures/" or "figures/" prefixes; keep just the filename.
  return (src:gsub("^%.%./figures/", ""):gsub("^figures/", ""))
end

function Image(el)
  el.src = strip_prefix(el.src)
  -- Pandoc only accepts CSS-like dimension values (e.g. "100%", "10cm").
  -- Raw LaTeX macros like "\linewidth" are filtered out by pandoc's image
  -- handling. Use "100%" which pandoc renders to "\linewidth" in LaTeX.
  if el.attributes["width"] == nil or el.attributes["width"] == "" then
    el.attributes["width"] = "100%"
  end
  return el
end

-- Box-drawing and other UTF-8 chars used in ASCII tree diagrams inside code
-- fences. LaTeX's `verbatim` environment is byte-literal and can't decode
-- multi-byte UTF-8 sequences, so we substitute ASCII equivalents *before*
-- the code block reaches LaTeX.
local box_drawing_replacements = {
  ["─"] = "-",
  ["│"] = "|",
  ["├"] = "+",
  ["└"] = "`",
  ["┌"] = "+",
  ["┐"] = "+",
  ["┘"] = "'",
  ["┤"] = "+",
  ["┬"] = "+",
  ["┴"] = "+",
  ["┼"] = "+",
  ["→"] = "->",
  ["←"] = "<-",
  ["↑"] = "^",
  ["↓"] = "v",
  ["≥"] = ">=",
  ["≤"] = "<=",
  ["≠"] = "!=",
  ["×"] = "x",
  ["…"] = "...",
}

local function fix_codeblock_text(s)
  for from, to in pairs(box_drawing_replacements) do
    s = s:gsub(from, to)
  end
  return s
end

function CodeBlock(el)
  el.text = fix_codeblock_text(el.text)
  return el
end

function Code(el)
  el.text = fix_codeblock_text(el.text)
  return el
end

-- Content-aware column widths for tables. Pandoc's default for pipe-table
-- output assigns equal widths to all columns; this looks terrible when one
-- column holds "Individual Test Accuracy (seeds 42, 123, 456, 789, 1024)"
-- and another holds "Std". Walk the table cells, measure the longest line
-- of text in each column, and rewrite colspecs so widths track content.
-- Each column is clamped to [MIN_W, MAX_W] of \linewidth so no column gets
-- so narrow it can't render a single value, nor so wide it eats the table.

local MIN_W = 0.07
local MAX_W = 0.42
local PAD_CHARS = 4

local function cell_max_line_length(cell)
  -- Stringify the cell's blocks, then split on newlines and find the
  -- longest line. Using the longest LINE (not total length) lets multi-
  -- paragraph cells still get a reasonable width.
  local text = pandoc.utils.stringify(cell.contents)
  local max_len = 0
  for line in (text .. "\n"):gmatch("([^\n]*)\n") do
    local len = #line
    if len > max_len then max_len = len end
  end
  return max_len
end

local function cell_max_unbreakable(cell)
  -- Length of the longest single word (LaTeX won't split it). Used as a
  -- character floor for column width so a header word like "Photometric"
  -- (11 chars, unbreakable) doesn't overflow into the next column.
  local text = pandoc.utils.stringify(cell.contents)
  local max_word = 0
  for word in text:gmatch("%S+") do
    if #word > max_word then max_word = #word end
  end
  return max_word
end

local function collect_rows(tbl)
  local rows = {}
  for _, r in ipairs(tbl.head.rows or {}) do
    table.insert(rows, r)
  end
  for _, body in ipairs(tbl.bodies or {}) do
    for _, r in ipairs(body.body or {}) do
      table.insert(rows, r)
    end
  end
  return rows
end

function Table(tbl)
  local ncols = #tbl.colspecs
  if ncols == 0 then return tbl end

  local rows = collect_rows(tbl)
  if #rows == 0 then return tbl end

  -- Per-column statistics:
  --   max_lens  = longest content line per column (drives proportional sizing)
  --   word_floor = longest unbreakable word per column (drives minimum width)
  local max_lens = {}
  local word_floor = {}
  for i = 1, ncols do max_lens[i] = 0; word_floor[i] = 0 end

  -- Use sqrt-weighted lengths so very long cells (e.g. a 100-char paragraph
  -- in one row) don't drown out other columns. Without dampening, one
  -- outlier sentence in a single cell makes that column dominate.
  local function dampened(n)
    -- Sqrt with a small floor; PAD_CHARS adjusted into the base.
    return math.sqrt(n + PAD_CHARS)
  end

  for _, row in ipairs(rows) do
    for i, cell in ipairs(row.cells) do
      if i <= ncols then
        local len = cell_max_line_length(cell)
        if len > max_lens[i] then max_lens[i] = len end
        local word = cell_max_unbreakable(cell)
        if word > word_floor[i] then word_floor[i] = word end
      end
    end
  end

  -- Compute proportional widths from sqrt-dampened max lengths.
  local total = 0
  for i = 1, ncols do total = total + dampened(max_lens[i]) end
  if total == 0 then return tbl end

  local widths = {}
  for i = 1, ncols do
    local w = dampened(max_lens[i]) / total
    -- Enforce a per-column floor: at least enough character-width to hold the
    -- longest unbreakable word. Approximate 1 char ≈ 0.55% of linewidth (10pt
    -- text on ~11cm linewidth ≈ 60 chars). Add 2 chars padding.
    local word_min = (word_floor[i] + 2) * 0.0095
    if w < word_min then w = word_min end
    if w < MIN_W then w = MIN_W end
    if w > MAX_W then w = MAX_W end
    widths[i] = w
  end

  local clamped_sum = 0
  for _, w in ipairs(widths) do clamped_sum = clamped_sum + w end
  for i = 1, ncols do widths[i] = widths[i] / clamped_sum end

  -- Rewrite colspecs preserving alignment, only changing width.
  local new_colspecs = {}
  for i = 1, ncols do
    new_colspecs[i] = {tbl.colspecs[i][1], widths[i]}
  end
  tbl.colspecs = new_colspecs
  return tbl
end

-- Caption promotion: pandoc renders `**Table N.M: ...**` and `**Figure N.M:** ...`
-- paragraphs as plain bold inline text, not as \caption{} calls. This means:
--   (a) the bold-text "caption" floats in awkward positions next to the table,
--   (b) figure captions wrap into the float box producing artefacts like
--       "Fig-" overflowing the page margin,
--   (c) \listoftables and \listoffigures find no captions and stay empty.
-- The Blocks-level pass below walks the document, detects bold "Table N.M:"
-- / "Figure N.M:" paragraphs immediately before a Table or after an Image
-- (Figure), strips the "Table N.M:" prefix (LaTeX will provide the
-- numbering), and attaches the remainder as a proper Caption.

local CAPTION_PREFIX_RE = "^(Table%s+%d+%.%d+[a-z]?:?%s*|Figure%s+%d+%.%d+[a-z]?:?%s*)"

local function strip_caption_prefix(inlines)
  -- Return inlines with leading "Table N.M:" / "Figure N.M:" removed.
  -- Returns nil if the inlines don't start with such a prefix.
  -- Also accepts optional letter suffix like "Table 7.9b:" so the rare
  -- supplementary-table convention works.
  if #inlines < 1 then return nil end
  local first = inlines[1]
  if first.t ~= "Strong" then return nil end
  local s = pandoc.utils.stringify(first.content)
  -- Look for the typed prefix.
  local prefix = s:match("^Table%s+%d+%.%d+[a-z]?:") or s:match("^Figure%s+%d+%.%d+[a-z]?:")
  if not prefix then return nil end

  -- Text after the prefix WITHIN the Strong (if any), keeping its content.
  local strong_tail = s:sub(#prefix + 1):gsub("^%s+", "")

  local out = {}
  if strong_tail ~= "" then
    table.insert(out, pandoc.Str(strong_tail))
  end
  -- Append everything after the first Strong. If the Strong had a tail
  -- AND there's more content after the original Strong, the boundary
  -- between the two needs a Space (since the original Strong absorbed
  -- the trailing punctuation but no trailing whitespace, and the next
  -- inline is typically the Space that was AFTER the Strong in the
  -- source). We keep that Space by NOT dropping it.
  for i = 2, #inlines do
    table.insert(out, inlines[i])
  end
  -- Edge case: if there was no Strong tail (strong was "Table 5.1:" only),
  -- the immediately-following Space is now leading whitespace; pandoc
  -- handles that fine when rendering.
  return out
end

local function is_caption_para(blk)
  if blk == nil or blk.t ~= "Para" then return nil end
  return strip_caption_prefix(blk.content)
end

local function try_split_image_caption_para(blk)
  -- Markdown like:
  --     ![Alt](img.png)
  --     **Figure N.M:** Caption text.
  -- (no blank line between) parses as ONE Para containing
  --     [Image, SoftBreak, Strong "Figure N.M:", Space, ...caption...]
  -- Pandoc's implicit_figures doesn't fire because the Image isn't alone.
  -- This helper detects that shape and returns (figure_block, nil) where
  -- figure_block is a proper pandoc.Figure with the bold caption attached.
  -- Returns nil if blk doesn't match the shape.
  if blk == nil or blk.t ~= "Para" or #blk.content < 3 then return nil end
  local inlines = blk.content
  if inlines[1].t ~= "Image" then return nil end
  -- After the Image we expect a SoftBreak or LineBreak, then a Strong with
  -- the caption prefix.
  local sep_idx = 2
  while sep_idx <= #inlines and (inlines[sep_idx].t == "SoftBreak" or inlines[sep_idx].t == "LineBreak" or inlines[sep_idx].t == "Space") do
    sep_idx = sep_idx + 1
  end
  if sep_idx > #inlines then return nil end
  -- Build a sub-list starting at sep_idx and try to strip the caption prefix.
  local tail = {}
  for i = sep_idx, #inlines do
    table.insert(tail, inlines[i])
  end
  local caption_inlines = strip_caption_prefix(tail)
  if not caption_inlines then return nil end
  -- Construct a Figure block: content is the Image wrapped in a Plain,
  -- caption is the stripped inlines as a Plain inside long.
  local img = inlines[1]
  return pandoc.Figure(
    { pandoc.Plain({ img }) },
    { long = pandoc.Blocks({ pandoc.Plain(caption_inlines) }) }
  )
end

function Pandoc(doc)
  local new_blocks = {}
  local i = 1
  while i <= #doc.blocks do
    local b = doc.blocks[i]
    local nxt = doc.blocks[i + 1]

    -- First check: is this a "[Image, SoftBreak, Strong 'Figure N.M:', ...]"
    -- mashed-into-one-Para case that pandoc didn't auto-figure-ify?
    local synthesised_figure = try_split_image_caption_para(b)
    if synthesised_figure then
      table.insert(new_blocks, synthesised_figure)
      i = i + 1
      goto continue
    end

    local caption_inlines = is_caption_para(b)

    if caption_inlines then
      if nxt and nxt.t == "Table" then
        -- Table caption: attach the bold-text paragraph as the table's caption.
        nxt.caption = {
          long = { pandoc.Plain(caption_inlines) },
        }
        table.insert(new_blocks, nxt)
        i = i + 2
      elseif #new_blocks > 0 and new_blocks[#new_blocks].t == "Figure" then
        -- Figure caption that appears AFTER the image (pandoc emits Figure
        -- block from `![Alt](img.png)` on its own line).
        local fig = new_blocks[#new_blocks]
        fig.caption = {
          long = { pandoc.Plain(caption_inlines) },
        }
        i = i + 1
      else
        -- Bold "Table/Figure N.M:" paragraph that isn't adjacent to a
        -- Table/Figure -- leave it as-is.
        table.insert(new_blocks, b)
        i = i + 1
      end
    else
      table.insert(new_blocks, b)
      i = i + 1
    end
    ::continue::
  end
  doc.blocks = new_blocks
  return doc
end

-- Prefix every heading's identifier with the chapter slug so labels like
-- `introduction` (which appears in chapters 1, 3, 4, 5) become unique:
-- `ch01-introduction`, `ch03-introduction`, etc. The prefix comes from the
-- THESIS_CH_PREFIX env var; if absent we leave identifiers untouched.
local id_prefix = os.getenv("THESIS_CH_PREFIX")
if id_prefix and id_prefix ~= "" then
  function Header(el)
    if el.identifier and el.identifier ~= "" then
      el.identifier = id_prefix .. el.identifier
    end
    return el
  end
end
