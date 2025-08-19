#include "connection.hpp"


Connection::Connection()
{
    this->value = 0.0;
}

Connection::~Connection()
{
}

void Connection::changeValue(double newValue)
{
    this->value = newValue;
}

double Connection::getValue()
{
    return this->value;
}
