#include <iostream>

#include "include/base_solver.hpp"

int main()
{
    std::cout << "Hello World!" << std::endl;

    variable<0> x;
    variable<1> y;
    variable<2> z;

    boost::array<double, 3> xv;
    boost::array<double, 3> result = newton_solver(std::make_tuple(x - y/**y + z*z - 14*/,
                                                                   x - y,
                                                                   x + y - z), xv, 100);

    return 0;
}
