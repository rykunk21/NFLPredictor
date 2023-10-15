import src.main
import sys

"""




"""

if __name__ == '__main__':
    
    # Get CL args
    args = sys.argv[1:]
    for i, arg in enumerate(sys.argv[1:]):
        print(f"Argument {i + 1}: {arg}")


    src.main.main(args)


