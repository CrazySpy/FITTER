#ifndef FCM_H
#define FCM_H

#include <Eigen/Dense>

class FCM{
private:

private:
    double _updateMembership(); // returns the max diff
    void _computeCenters();
    double _calculateDistance(unsigned int i, unsigned int k);
    double _calculatePointMembership(unsigned int i, unsigned int k);

public:
    FCM(double, double);
    ~FCM();

    void setData(Eigen::MatrixXd *);
    void setMembership(Eigen::MatrixXd *);
    void initMembership();
    void setClusterNum(unsigned int);

    Eigen::MatrixXd * getData();
    Eigen::MatrixXd * getMembership();
    Eigen::MatrixXd * getClusterCenters();

    void execute();

private:
    double _m; // the fuzziness
    double _epsilon; // threshold to stop
    unsigned int _clusterNum;
    unsigned int _dimensionNum;

    Eigen::MatrixXd * _membershipMatrix;
    Eigen::MatrixXd * _dataMatrix;
    Eigen::MatrixXd * _clusterCenters;

};

#endif

