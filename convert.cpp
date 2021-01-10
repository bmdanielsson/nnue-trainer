#include <iostream>
#include <string>
#include <cstddef>

#include "lib/nnue_training_data_formats.h"

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cout << "convert <input file> <output file>" << std::endl;
        std::cout << std::endl;
        std::cout << "Conversion is done based on file extension." << std::endl;
        std::cout << "Supported extensions: plain, bin, binpack" << std::endl;
        return 1;
    }

    std::string input(argv[1]);
    std::size_t found = input.find_last_of(".");
    if (found == std::string::npos) {
        std::cout << "Filename extension missing from input file" << std::endl;
        return 1;
    }
    std::string inExt = input.substr(found);

    std::string output(argv[2]);
    found = output.find_last_of(".");
    if (found == std::string::npos) {
        std::cout << "Filename extension missing from output file" << std::endl;
        return 1;
    }
    std::string outExt = output.substr(found);

    if (inExt == ".plain" && outExt == ".bin") {
        binpack::convertPlainToBin(input, output, std::ios_base::app, false);
    } else if (inExt == ".plain" && outExt == ".binpack") {
        binpack::convertPlainToBinpack(input, output, std::ios_base::app, false);
    } else if (inExt == ".bin" && outExt == ".plain") {
        binpack::convertBinToPlain(input, output, std::ios_base::app, false);
    } else if (inExt == ".bin" && outExt == ".binpack") {
        binpack::convertBinToBinpack(input, output, std::ios_base::app, false);
    } else if (inExt == ".binpack" && outExt == ".plain") {
        binpack::convertBinpackToPlain(input, output, std::ios_base::app, false);
    } else if (inExt == ".binpack" && outExt == ".bin") {
        binpack::convertBinpackToBin(input, output, std::ios_base::app, false);
    } else {
        std::cout << "Unsupported conversion" << std::endl;
        return 1;
    }

    return 0;
}
