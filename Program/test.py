import os

src_dir = '../Data/Datasets/'
files = os.walk(src_dir).next()[2]

current_dir = os.path.dirname(__file__)
file_path = '../Data/{}'.format("Combined_datasets_ratings.txt")
file_rel_path = os.path.join(current_dir, file_path)
benchmarks_file = open(file_rel_path, 'r')
benchmarks = [ b.replace("\r\n", "").split(";")[0] for b in benchmarks_file ]
benchmarks_file.close()

not_found = []
for f in files:
    if f not in benchmarks:
        not_found.append(f)

print not_found