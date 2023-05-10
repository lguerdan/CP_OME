import os, json

def write_file(data, filepath, filename):
    if os.path.isdir(filepath) == False:
        try:
            os.makedirs(filepath)
        except:
            print('Error making directory')

    path = os.path.join(filepath, filename)

    if '.csv' in filename:
        data.to_csv(path)
    else:
        with open(path, "w") as outfile:
            outfile.write(data)
