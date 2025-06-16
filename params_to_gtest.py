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

PARAMS_PATH = "params_w_bias.txt"
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
        # --- Robust field extraction using /*fieldName=*/ markers ---
        import re
        field_names = [
            'input', 'originalWeightsShape', 'weight', 'in_channels', 'out_channels', 'batch_size', 'input_height', 'input_width',
            'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'conv2dConfig', 'outputMemoryConfig'
        ]
        fields = {}
        for name in field_names:
            pat = rf'/\*{name}=*\*/(.*?)(?=/\*|$)'
            m = re.search(pat, args_str, re.DOTALL)
            if m:
                fields[name] = m.group(1).strip(' ,\n')
            else:
                fields[name] = ''
        # Apply C++ fixes where needed
        fields['input'] = fix_grid_for_cpp(fix_shape_for_cpp(fields['input']))
        fields['weight'] = fix_grid_for_cpp(fix_shape_for_cpp(fields['weight']))
        fields['outputMemoryConfig'] = fix_grid_for_cpp(fields['outputMemoryConfig'])
        # Print debug for each field
        for k, v in fields.items():
            print(f"[DEBUG] {k}: {v}")
        return {
            'input_spec': fields['input'],
            'original_weights_shape': fields['originalWeightsShape'],
            'weight_spec': fields['weight'],
            'in_channels': fields['in_channels'],
            'out_channels': fields['out_channels'],
            'batch_size': fields['batch_size'],
            'input_height': fields['input_height'],
            'input_width': fields['input_width'],
            'kernel_size': fields['kernel_size'],
            'stride': fields['stride'],
            'padding': fields['padding'],
            'dilation': fields['dilation'],
            'groups': fields['groups'],
            'bias': fields['bias'],
            'conv2d_config': fields['conv2dConfig'],
            'mem_config': fields['outputMemoryConfig'],
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
    s = s.strip()
    # If already wrapped in braces, return as is
    if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
        return s
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
    return f"  {{\n    /* input_spec = */ {args_parsed['input_spec']},\n    /* original_weights_shape = */ {parse_vec(args_parsed['original_weights_shape'])},\n    /* weight_spec = */ {args_parsed['weight_spec']},\n    /* in_channels = */ {args_parsed['in_channels']},\n    /* out_channels = */ {args_parsed['out_channels']},\n    /* batch_size = */ {args_parsed['batch_size']},\n    /* input_height = */ {args_parsed['input_height']},\n    /* input_width = */ {args_parsed['input_width']},\n    /* kernel_size = */ {parse_vec(args_parsed['kernel_size'])},\n    /* stride = */ {parse_vec(args_parsed['stride'])},\n    /* padding = */ {parse_vec(args_parsed['padding'])},\n    /* dilation = */ {parse_vec(args_parsed['dilation'])},\n    /* groups = */ {args_parsed['groups']},\n    /* bias = */ {args_parsed['bias']},\n    /* conv2d_config = */ {args_parsed['conv2d_config']},\n    /* output_mem_config = */ {args_parsed['mem_config']}\n  }}"

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

