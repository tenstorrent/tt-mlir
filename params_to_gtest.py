import re

def fix_grid_for_cpp(mem_config_str):
    # Find grid=... even if there are newlines or spaces
    def grid_repl(match):
        grid_content = match.group(1)
        # Find all ranges of the form [(x=a,y=b) - (x=c,y=d)]
        pairs = []
        for rng in re.finditer(r'\[\(x=(\d+),y=(\d+)\)\s*-\s*\(x=(\d+),y=(\d+)\)\]', grid_content):
            a, b, c, d = rng.groups()
            pairs.append(f'{{{{{a}, {b}}}, {{{c}, {d}}}}}')
        return 'grid={' + ','.join(pairs) + '}'
    # Substitute grid=... with fixed C++ syntax, allowing for newlines and spaces
    return re.sub(r'grid=\{([^\}]*)\}', grid_repl, mem_config_str, flags=re.DOTALL)

def fix_shape_for_cpp(s):
    # Replace Shape([a, b, c, d]) with Shape({a, b, c, d})
    return re.sub(r'Shape\(\[([^\]]+)\]\)', r'Shape({\1})', s)

PARAMS_PATH = "params.txt"
GTEST_OUT_PATH = "params_gtest.inc"

# Regex to extract parameters from the log line
# Step 1: Extract argument list inside query_op_constraints(...)
paren_pattern = re.compile(r"::ttnn::graph::query_op_constraints\((.*)\)$")

def extract_args(line):
    m = paren_pattern.search(line.strip())
    if not m:
        return None
    args_str = m.group(1)
    # Remove leading device ptr and possible whitespace
    args_str = re.sub(r"^[^,]+,\s*", "", args_str)
    return args_str

# Step 2: Split arguments using markers and nested structure
import itertools

def parse_args(args_str):
    # Find /*input*/TensorSpec(...), /*weight*/TensorSpec(...)
    # Then, parse the rest by commas, handling nested parens
    try:
        # Marker-based extraction for robustness
        input_marker = '/*input*/TensorSpec'
        weight_marker = '/*weight*/TensorSpec'
        input_start = args_str.index(input_marker)
        weight_start = args_str.index(weight_marker)
        print(f"[DEBUG] input_marker index: {input_start}")
        print(f"[DEBUG] weight_marker index: {weight_start}")
        input_spec = args_str[input_start : weight_start].rstrip(', ')
        input_spec = fix_grid_for_cpp(input_spec)
        input_spec = fix_shape_for_cpp(input_spec)
        print(f"[DEBUG] input_spec: {input_spec}")
        # For weight_spec, find closing paren after weight_marker
        weight_tensor_idx = weight_start + len('/*weight*/')
        weight_paren_start = args_str.index('(', weight_tensor_idx)
        print(f"[DEBUG] weight_paren_start: {weight_paren_start}")
        def find_matching_paren(s, start):
            count = 0
            for i, c in enumerate(s[start:], start):
                if c == '(': count += 1
                elif c == ')': count -= 1
                if count == 0: return i+1
            return len(s)  # If not found, return end of string
        # New logic: take everything from weight_start up to 'in_channels=' as weight_spec
        in_channels_idx = args_str.find('in_channels=', weight_start)
        if in_channels_idx == -1:
            print('[ERROR] Could not find in_channels= after weight_spec!')
            weight_spec = args_str[weight_start:]
            rest = ''
        else:
            weight_spec = args_str[weight_start:in_channels_idx].rstrip(', ')
        weight_spec = fix_grid_for_cpp(weight_spec)
        weight_spec = fix_shape_for_cpp(weight_spec)
        rest = args_str[in_channels_idx:]
        print(f"[DEBUG] weight_spec: {weight_spec}")
        rest = rest.lstrip(', \n')
        # Now split the rest by commas, but handle nested parens/brackets
        def smart_split(s):
            out = []
            buf = ''
            nest = 0
            for c in s:
                if c in '([{' : nest += 1
                if c in ')]}': nest -= 1
                if c == ',' and nest == 0:
                    out.append(buf.strip())
                    buf = ''
                else:
                    buf += c
            if buf.strip(): out.append(buf.strip())
            return out
        rest_args = smart_split(rest)
        print("[DEBUG] rest_args after smart_split:")
        print(rest_args)  # Print first two rest_args
        # Map by expected order
        # [in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, bias, Conv2dConfig, MemoryConfig]
        if len(rest_args) < 13:
            print(f"[ERROR] Not enough arguments in rest_args (got {len(rest_args)}): {rest_args}")
            return None
        in_channels = rest_args[0].split('=')[-1].strip()
        out_channels = rest_args[1].split('=')[-1].strip()
        batch_size = rest_args[2].split('=')[-1].strip()
        input_height = rest_args[3].split('=')[-1].strip()
        input_width = rest_args[4].split('=')[-1].strip()
        kernel_size = rest_args[5].split('=')[-1].strip()
        stride = rest_args[6].split('=')[-1].strip()
        padding = rest_args[7].split('=')[-1].strip()
        dilation = rest_args[8].split('=')[-1].strip()
        groups = rest_args[9].strip()
        bias = rest_args[10].replace('/*bias*/', '').strip()
        conv2d_config = rest_args[11].strip()
        mem_config = rest_args[12].replace('/*outputMemoryConfig*/', '').strip()
        # --- Fix grid for C++ initializer syntax ---
        mem_config = fix_grid_for_cpp(mem_config)
        print(f"[DEBUG] in_channels: {in_channels}")
        print(f"[DEBUG] out_channels: {out_channels}")
        print(f"[DEBUG] batch_size: {batch_size}")
        print(f"[DEBUG] input_height: {input_height}")
        print(f"[DEBUG] input_width: {input_width}")
        print(f"[DEBUG] kernel_size: {kernel_size}")
        print(f"[DEBUG] stride: {stride}")
        print(f"[DEBUG] padding: {padding}")
        print(f"[DEBUG] dilation: {dilation}")
        print(f"[DEBUG] groups: {groups}")
        print(f"[DEBUG] bias: {bias}")
        print(f"[DEBUG] conv2d_config: {conv2d_config}")
        print(f"[DEBUG] mem_config: {mem_config}")
        batch_size = rest_args[2].split('=')[-1].strip()
        input_height = rest_args[3].split('=')[-1].strip()
        input_width = rest_args[4].split('=')[-1].strip()
        kernel_size = rest_args[5].split('=')[-1].strip('[] ')
        stride = rest_args[6].split('=')[-1].strip('[] ')
        padding = rest_args[7].split('=')[-1].strip('[] ')
        dilation = rest_args[8].split('=')[-1].strip('[] ')
        groups = rest_args[9]
        bias = rest_args[10].replace('/*bias*/', '').strip()
        conv2d_config = rest_args[11].strip()
        return {
            'input_spec': input_spec,
            'weight_spec': weight_spec,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'batch_size': batch_size,
            'input_height': input_height,
            'input_width': input_width,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias,
            'conv2d_config': conv2d_config,
            'mem_config': mem_config
        }
    except Exception as e:
        # Print debug info for the first few failures
        if not hasattr(parse_args, "fail_count"):
            parse_args.fail_count = 0
        if parse_args.fail_count < 3:
            print("\n==== DEBUG FAILURE SAMPLE ====")
            print(f"Exception: {e}")
            print(f"Full args_str:\n{args_str}")
            print(f"rest_args (length {len(locals().get('rest_args', []))}):\n{locals().get('rest_args', [])}")
            print("============================\n")
            parse_args.fail_count += 1
        print(f"[ERROR] Failed to parse args: {e}\n{args_str[:200]}...")
        return None

def parse_vec(s):
    # Converts '7, 7' to '{7, 7}'
    return '{' + ', '.join([x.strip() for x in s.split(',')]) + '}'

def parse_line(line):
    args = extract_args(line)
    if not args:
        print("[DEBUG] extract_args returned None for line:")
        print(line)
        return None
    print("[DEBUG] args_str after extract_args:")
    print(args)
    args_parsed = parse_args(args)
    if not args_parsed:
        return None
    return f"  {{\n    /* input_spec = */ {args_parsed['input_spec']},\n    /* weight_spec = */ {args_parsed['weight_spec']},\n    /* in_channels = */ {args_parsed['in_channels']},\n    /* out_channels = */ {args_parsed['out_channels']},\n    /* batch_size = */ {args_parsed['batch_size']},\n    /* input_height = */ {args_parsed['input_height']},\n    /* input_width = */ {args_parsed['input_width']},\n    /* kernel_size = */ {parse_vec(args_parsed['kernel_size'])},\n    /* stride = */ {parse_vec(args_parsed['stride'])},\n    /* padding = */ {parse_vec(args_parsed['padding'])},\n    /* dilation = */ {parse_vec(args_parsed['dilation'])},\n    /* groups = */ {args_parsed['groups']},\n    /* bias = */ {args_parsed['bias']},\n    /* conv2d_config = */ {args_parsed['conv2d_config']},\n    /* output_mem_config = */ {args_parsed['mem_config']}\n  }}"

def main():
    with open(PARAMS_PATH, "r") as f:
        lines = f.readlines()
    blocks = []
    matched = 0
    unmatched = 0
    for idx, line in enumerate(lines):
        block = parse_line(line)
        if block:
            blocks.append(block)
            matched += 1
        else:
            print(f"[WARN] Line {idx+1} did not match:\n{line[:200]}...\n")
            unmatched += 1
    with open(GTEST_OUT_PATH, "w") as f:
        f.write(",\n".join(blocks))
        f.write("\n")
    print(f"[INFO] Matched {matched} lines, unmatched {unmatched} lines.")

if __name__ == "__main__":
    main()

