import argparse
import hashlib
import os

CHUNK_SIZE = 65536

def get_sha1(file_path):
    sha1  = hashlib.sha1()
    with open(args.source, 'rb') as file:
        while True:
            data = file.read(CHUNK_SIZE)
            if not data:
                break
            sha1.update(data)
      
    return sha1.hexdigest()


def main(args):
    sha1 = get_sha1(args.source)
    dirname = os.path.dirname(args.source)
    if dirname:
        new_file_path = dirname + '/' + f'net-{sha1[:7]}.nnue'
    else:
        new_file_path = f'net-{sha1[:7]}.nnue'
      
    print(f'SHA1: {sha1}')
    print(f'Name: {new_file_path}')
    os.rename(args.source, new_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename a model file to a more unique format.')
    parser.add_argument('source', help='Source file')
    args = parser.parse_args()
      
    main(args)
