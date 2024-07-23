#include <math.h>
#include "OptionCalculator.hpp"

// ****************************************************************
// ****************************************************************
OptionCalculator::OptionCalculator ( double T,
                                     double K,
                                     double S,
                                     double r,
                                     double d,
                                     double sigma)
{
    m_T = T/365; // Convert to daycount
    m_K = K;
    m_S = S;
    m_r = r;
    m_d = d;
    m_sigma = sigma;

    m_pi= std::atan(1.0)*4;
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::DiscountFactor()
{
    return 1 / (1 + m_r * m_T);
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::ForwardApproximation()
{
    return m_S * (1 + m_r * m_T) * exp(-m_d * m_T);
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::d1()
{
    return (log(ForwardApproximation() / m_K) + pow(m_sigma, 2) * m_T / 2) / (m_sigma * sqrt(m_T));
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::d2()
{
    return d1() - m_sigma * sqrt(m_T);
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::NormalCDFInverseDerivative ( double d )
{
    return (1 / sqrt(2 * m_pi)) * exp((pow(d, 2)) / 2);
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::NormalCDFInverse(double d)
{
    double n, k;

    k = 1 / (1 + m_gamma * d);
    n = 1 - NormalCDFInverseDerivative(d) * (m_a1 * k + m_a2 * pow(k, 2) + m_a3 * pow(k, 3) + m_a4 * pow(k, 4) + m_a5 * pow(k, 5));

    if (d < 0)
    {
        n = 1 - NormalCDFInverse(-d);
    }

    return n;
}

// ****************************************************************
// ****************************************************************
double OptionCalculator::CalculateCallOptionPrice()
{
    return DiscountFactor() * ((ForwardApproximation() * NormalCDFInverse(d1())) - (m_K * NormalCDFInverse(d2())));
}

// ****************************************************************
// ****************************************************************
double OptionCalculator :: CalculatePutOptionPrice()
{
    return DiscountFactor() * ((m_K * NormalCDFInverse(-d2())) - (ForwardApproximation() * NormalCDFInverse(-d1())));
}