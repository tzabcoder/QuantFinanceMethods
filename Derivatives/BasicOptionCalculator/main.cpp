#include <iostream>
#include "OptionCalculator.hpp"

int main ( int argc, char* argv[])
{
    if ( argc == 7 )
    {
        double S     = std::atof ( argv[1] );
        double K     = std::atof ( argv[2] );
        double r     = std::atof ( argv[3] );
        double d     = std::atof ( argv[4] );
        double sigma = std::atof ( argv[5] );
        double T     = std::atof ( argv[6] );

        std::cout << "Stock Price (S):        " <<  S << std::endl;
        std::cout << "Strike Price (K):       " <<  K << std::endl;
        std::cout << "Risk-Free Rate (r):     " <<  r << std::endl;
        std::cout << "Dividend Rate (d):      " <<  d << std::endl;
        std::cout << "Volatility (sigma):     " <<  sigma << std::endl;
        std::cout << "Days to Expiration (T): " <<  T << std::endl << std::endl;

        OptionCalculator calculator = OptionCalculator(T, K, S, r, d, sigma);

        double putPrice = calculator.CalculateCallOptionPrice();
        double callPrice = calculator.CalculatePutOptionPrice();

        std::cout << "Put Option Price:  $" << putPrice << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "Call Option Price: $" << callPrice << std::endl;
    }
    else
    {
        std::cout << "Need correct number of params..." << std::endl;
        return 1;
    }

    return 0;
}