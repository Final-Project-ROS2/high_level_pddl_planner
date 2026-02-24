import sys, os, datetime, json
from jinja2 import Template, Environment
from typing import Dict, List

def tif_filter(time: float, value: float, *function_name) -> str:
    """ Creates time-initial fluent if time>0, or plain initialization otherwise """
    assignment = "(= ({}) {})".format(' '.join(function_name), value)
    return "(at {} {})".format(time, assignment) if time > 0\
        else assignment

def load_template_from_string(template_text: str) -> Template:
    jinja2_env = Environment(trim_blocks = False, lstrip_blocks = False)
    jinja2_env.filters['tif'] = tif_filter

    return jinja2_env.from_string(template_text)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def remove_doubled_whitespace(text: str) -> str:
    lines = text.splitlines(True)

    output = ''
    was_white_space = False

    for line in lines:
        is_white_space = len(line.strip()) == 0
        if not is_white_space:
            output = output + line
        elif not was_white_space:
            output = output + line
        
        was_white_space = is_white_space

    return output

def transform(template_string: str, data: Dict) -> str:
    """ transforms the template with given data; this function may be called from other Python code """

    template = load_template_from_string(template_string)
    transformed = template.render(data=data)
    compacted = remove_doubled_whitespace(transformed)
    return compacted

def main(args):
    """ transforms the problem file template """
    if len(args) != 3:
        # print errors to the error stream
        eprint("Usage: {0} <template-file> <data-json-file> <output-file>".format(os.path.basename(sys.argv[0])))
        exit(-1)

    template_file = args[0]
    data_file = args[1]
    output_file = args[2]

    # Read template from file
    try:
        with open(template_file, 'r') as f:
            template_string = f.read()
    except Exception as e:
        eprint("Error reading template file {}: {}".format(template_file, e))
        exit(-1)

    # Read data from JSON file
    try:
        with open(data_file, 'r') as f:
            json_content = json.load(f)
    except Exception as e:
        eprint("Error reading data file {}: {}".format(data_file, e))
        exit(-1)

    # Extract the 'data' key if it exists, otherwise use the whole JSON
    data = json_content.get('data', json_content)

    # Render template with data
    try:
        transformed = transform(template_string, data)
    except Exception as e:
        eprint("Error rendering template: {}".format(e))
        exit(-1)

    # Write output to file
    try:
        with open(output_file, 'w') as f:
            f.write(transformed)
            f.write("\n; This PDDL problem file was generated on {}\n".format(str(datetime.datetime.now())))
    except Exception as e:
        eprint("Error writing output file {}: {}".format(output_file, e))
        exit(-1)

    print("Problem file generated successfully: {}".format(output_file))

if __name__ == "__main__":
    main(sys.argv[1:])