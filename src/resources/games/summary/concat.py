import os

def concatenate_files(subdir='W1', output_file='netsummaries.txt'):
    # Ensure the given subdir exists
    if not os.path.exists(subdir):
        print(f"Error: The directory '{subdir}' does not exist.")
        return

    # Get a list of all text files in the subdir
    files = [f for f in os.listdir(subdir) if f.endswith('.txt')]

    # If no text files are found
    if not files:
        print("No text files found in the specified directory.")
        return

    # Create or overwrite the output file
    with open(output_file, 'w') as outfile:
        for file in files:
            with open(os.path.join(subdir, file), 'r') as infile:
                # Read the content of the file and write to outfile
                outfile.write(infile.read())
                # Add a newline to ensure separation between files
                outfile.write('\n')
                
    print(f"Concatenation complete. Check the output in '{output_file}'.")

# Example usage
concatenate_files()
