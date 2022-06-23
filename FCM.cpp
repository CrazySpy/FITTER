#include <iostream>
#include <cmath>
#include <cfloat>

#include <Eigen/Dense>

#include "FCM.h"


FCM::FCM(double m, double epsilon)
    : _epsilon(epsilon),
      _m(m),
      _membershipMatrix(nullptr),
      _dataMatrix(nullptr),
      _clusterCenters(nullptr),
      _clusterNum(0),
      _dimensionNum(0) {

}

FCM::~FCM(){
    if(_membershipMatrix != nullptr){
        delete _membershipMatrix;
        _membershipMatrix = nullptr;
    }

    if(_clusterCenters != nullptr){
        delete _clusterCenters;
        _clusterCenters = nullptr;
    }
}


double FCM::_updateMembership(){
    if(_dataMatrix == nullptr || _dataMatrix->rows() == 0){
        throw std::logic_error("ERROR: data should not be empty when updating the membership");
    }

    if(_membershipMatrix == nullptr || _membershipMatrix->rows() == 0 || _membershipMatrix->rows() != _dataMatrix->rows()){
        //cout << "init the membership";
        this->initMembership();
    }
    if(_clusterNum == 0){
        throw std::logic_error("ERROR: the number of clusters should be set");
    }

//    cout <<"mdata rows: "<< _dataMatrix->rows()<<endl;
//    cout << "mdata cols: "<<_dataMatrix->cols()<<endl;

    double new_uik;
    double max_diff = 0.0, diff;
    for (int i = 0; i < _clusterNum; i++) {
        for (int k = 0; k < _dataMatrix->rows(); k++) {
            //cout << "point: " << k << " and cluster" << i <<endl;
            //cout << "\nwill ask for the new new_uik"<< endl;
            new_uik = this->_calculatePointMembership(i, k);
            //cout << new_uik << endl;
            diff = new_uik - (*_membershipMatrix)(k, i);
            if (diff > max_diff){
                max_diff = diff;
            }
            (*_membershipMatrix)(k, i) = new_uik;
        }
    }

    return max_diff;
}


void FCM::_computeCenters(){
    if(_dataMatrix == nullptr || _dataMatrix->rows() == 0){
        throw std::logic_error("ERROR: number of rows is zero");
        return;
    }

    Eigen::MatrixXd t(_dataMatrix->rows(), _clusterNum);
    for (int i = 0; i < _dataMatrix->rows(); i++) { // compute (u^m) for each cluster for each point
        for (int j = 0; j < _clusterNum; j++) {
            t(i,j) = pow((*_membershipMatrix)(i, j), _m);
        }
    }

    for(int j = 0; j < _clusterNum; ++j) {
        Eigen::RowVectorXd numerator = Eigen::VectorXd::Zero(_dimensionNum);
        for(int i = 0; i < _dataMatrix->rows(); ++i) {
            numerator += t(i, j) * (*_dataMatrix)(i, Eigen::indexing::all);
        }
        double denominator = (*_membershipMatrix)(Eigen::indexing::all, j).sum();

        (*_clusterCenters)(j, Eigen::indexing::all) = numerator / denominator;
    }

//    for (int j = 0; j < _clusterNum; j++) { // loop for each cluster
//        for (int k = 0; k < _dimensionNum; k++) { // for each dimension
//            double numerator = 0.0, denominator = 0.0;
//            for (int i = 0; i < _dataMatrix->rows(); i++) {
//                numerator += t(i,j) * (*_dataMatrix)(i, k);
//                denominator += t(i,j);
//            }
//            (*_clusterCenters)(j, k) = numerator / denominator;
//        }
//    }
}

double FCM::_calculateDistance(unsigned int i, unsigned int k){
  /*
   * distance which is denoted in the paper as d
   * k is the data point
   * i is the cluster center point
  */
  //cout<<"_calculateDistance: point: "<<k<<" and cluster "<<i<<endl;
  if(_clusterNum == 0){
      throw std::logic_error("ERROR: number of clusters should not be zero\n");
  }
  if(_dimensionNum == 0){
      throw std::logic_error("ERROR: number of dimensions should not be zero\n");
  }

  return ((*_dataMatrix)(k, Eigen::indexing::all) - (*_clusterCenters)(i, Eigen::indexing::all)).lpNorm<2>();
}

double FCM::_calculatePointMembership(unsigned int i, unsigned int k){
    /*
     * i the cluster
     * k is the data point
    */
    //cout << __func__ <<"  num of cluster: "<<_clusterNum<<endl;
    double t, seg=0.0;
    double exp = 2 / (_m - 1);
    double dik, djk;
    if(_clusterNum == 0){
        throw std::logic_error("ERROR: number of clusters should not be zero\n");
    }
    for (unsigned int j = 0; j < _clusterNum; j++) {
      dik = this->_calculateDistance(i, k);
      djk = this->_calculateDistance(j, k);
      if(djk==0){
          djk = DBL_MIN;
      }
      t = dik / djk;
      t = pow(t, exp);
      //cout << "cluster: " << i << "data: " << k << " - " << "t: "<<t<<endl;
      seg += t;
    }
    //cout << "seg: "<<seg << " u: "<<(1.0/seg)<<endl;
    return 1.0 / seg;
}


void FCM::setData(Eigen::MatrixXd *data){
    if(_dataMatrix != nullptr){
        delete _dataMatrix;
    }
    if(data->rows()==0){
        throw std::logic_error("ERROR: seting empty data");
    }
    _dataMatrix = data;
    _dimensionNum = _dataMatrix->cols();
}

void FCM::setMembership(Eigen::MatrixXd *membership){
    if(_dataMatrix == 0){
        throw std::logic_error("ERROR: the data should present before setting up the membership");
    }
    if(_clusterNum == 0){
        if(membership->cols() == 0){
            throw std::logic_error("ERROR: the number of clusters is 0 and the membership matrix is empty");
        }
        else{
            this->setClusterNum(membership->cols());
        }
    }
    if(_membershipMatrix != nullptr){
        delete _membershipMatrix;
    }

    _membershipMatrix = membership;
    if(_membershipMatrix->rows() == 0){
        _membershipMatrix->resize(_dataMatrix->rows(), _clusterNum);
    }
}

void FCM::initMembership(){
    if(_clusterNum == 0){
        throw std::logic_error("ERROR: the number of clusters is 0");
    }
    if(_dataMatrix == nullptr){
        throw std::logic_error("ERROR: the data should present before setting up the membership");
    }
    if(_membershipMatrix != nullptr){
        delete _membershipMatrix;
    }

    _membershipMatrix = new Eigen::MatrixXd;
    _membershipMatrix->resize(_dataMatrix->rows(), _clusterNum);
    _updateMembership();
//    double mem = 1.0 / _clusterNum;
//    for(int j=0; j < _clusterNum; j++){
//        for(int i=0; i < _dataMatrix->rows(); i++){
//            (*_membershipMatrix)(i, j) = mem;
//        }
//    }
}

void FCM::setClusterNum(unsigned int ClusterNum){
    _clusterNum = ClusterNum;
    if(_clusterCenters){
        delete _clusterCenters;
    }
    _clusterCenters = new Eigen::MatrixXd;
    *_clusterCenters = Eigen::MatrixXd::Random(_clusterNum, _dimensionNum);
}

Eigen::MatrixXd * FCM::getData(){
    return _dataMatrix;
}

Eigen::MatrixXd * FCM::getMembership(){
    return _membershipMatrix;
}

Eigen::MatrixXd * FCM::getClusterCenters(){
    return _clusterCenters;
}

void FCM::execute() {
    while(true) {
        _computeCenters();
        double maxDiff = _updateMembership();
//        std::cout << *_membershipMatrix << std::endl;
//        std::cout << *_clusterCenters << std::endl;
//        std::cout << maxDiff << std::endl;
        if(maxDiff < _epsilon) {
            break;
        }
    }
}









