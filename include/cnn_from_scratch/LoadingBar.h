#pragma once

#include <iostream>
#include <iomanip>

void loadingBar(float val, float total, size_t width = 50){
    size_t occupied_width = val/total * width;
    size_t unoccupied_width = width - occupied_width;
    printf("\033[?25l");
    std::cout << "\r" << std::setfill('=') << std::setw(occupied_width+1) << std::left << "[";
    std::cout << std::setfill(' ') << std::setw(unoccupied_width) << std::right << "]";
    std::cout << " " << std::setw(3) << (int)(100.0*val/total+1) << "%";
    printf("\033[?25h");
}