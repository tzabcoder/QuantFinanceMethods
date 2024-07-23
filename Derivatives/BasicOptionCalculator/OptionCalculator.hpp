#include <iostream>
#include <cmath>

#ifndef OPTIONCALCULATOR_H_
#define OPTIONCALCULATOR_H_

/*************************************************************
* @class OptionCalculator
*
* Description:
* This class implements a basic version of an option value calculator
* using the Black-Scholes model. The calculator uses a standard
* normal distribution approximation.
*************************************************************/
class OptionCalculator
{
public:
    /*************************************************************
     * Constructor for OptionCalculator()
     ************************************************************/
    OptionCalculator ( double T,
                       double K,
                       double S,
                       double r,
                       double d,
                       double sigma );

    /*************************************************************
     * CalculateCallOptionPrice()
     *
     * Calculates the Call option price from the input parameters.
     ************************************************************/
    double CalculateCallOptionPrice();

    /*************************************************************
     * CalculatePutOptionPrice()
     *
     * Calculates the Put option price from the input parameters.
     ************************************************************/
    double CalculatePutOptionPrice();

private:
    /*************************************************************
     * DiscountFactor()
     *
     * Estimates a simplified discount factor using only one interest
     * rate (the discount rate). Given:
     * r - the interest rate
     * T - maturity of the option
     * The discount factor is estimated as:
     *
     * df(0,T) = 1/(1+r*T)
     ************************************************************/
    double DiscountFactor();

    /*************************************************************
     * ForwardApproximation()
     *
     * Estimates the forward rate for the stock price.
     * Given:
     * S - spot stock price
     * r - risk-free rate
     * T - time to maturity
     * d - dividend rate
     * The forward rate for stocks is approximated as:
     *
     * Ft = S*(1+r*T)*e^(-d*T)
     ************************************************************/
    double ForwardApproximation();

    /*************************************************************
     * d1()
     *
     * Calculates d1
     ************************************************************/
    double d1();

    /*************************************************************
     * d1()
     *
     * Calculates d2
     ************************************************************/
    double d2();

    /*************************************************************
     * NormalCDFInverse()
     *
     * Calculates N(d)
     * @param[in] d The input to the normal CDF
     ************************************************************/
    double NormalCDFInverse ( double d );

    /*************************************************************
     * NormalCDFInverseDerivative()
     *
     * Calculates N'(d)
     * @param[in] d The input to the normal CDF
     ************************************************************/
    double NormalCDFInverseDerivative ( double d );

    //! Normal distribution approximations
    const double m_gamma = 0.2316419;
    const double m_a1    = 0.319381530;
    const double m_a2    = -0.356563782;
    const double m_a3    = 1.781477937;
    const double m_a4    = -1.821255978;
    const double m_a5    = 1.330274429;

    //! Option calculation variables
    double m_T;     // Time to expiration
    double m_K;     // Underlying strike price
    double m_S;     // Underlying stock price
    double m_r;     // Discount rate
    double m_d;     // Continously compounded dividend rate
    double m_sigma; // Standard deviation (volatility) of underlying

    double m_pi;
};

#endif // OPTIONCALCULATOR_H_
