import src.main
import sys

"""


"""

def parse_args(argv):
    """
    Parses the arguments
    """
    args_dict = {}
    key = None
    for arg in argv[1:]:  # skip the script name
        if arg.startswith('-'):
            if key:
                args_dict[key] = tuple(values)
            key = arg[1:]
            values = []
        else:
            values.append(arg)
    if key:
        args_dict[key] = tuple(values)
    return args_dict



if __name__ == '__main__':
    
    # Get CL args
    parsed_args = parse_args(sys.argv)
    src.main.main(**parsed_args)


