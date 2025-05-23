#ifndef CONNECTION_HPP
#define CONNECTION_HPP

class Connection {
public:
    Connection();
    ~Connection();

    void changeValue(double newValue);
    double getValue();

private:
    double value;
};

#endif // CONNECTION_HPP
