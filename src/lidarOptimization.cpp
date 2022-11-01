// Author of ISCLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "lidarOptimization.h"
#include <unordered_set>
std::map<int, int> label_map = {
    {0, 0},
    {1, 0},
    {10, 1},
    {11, 2},
    {13, 5},
    {15, 3},
    {16, 5},
    {18, 4},
    {20, 5},
    {30, 6},
    {31, 7},
    {32, 8},
    {40, 9},
    {44, 10},
    {48, 11},
    {50, 13},
    {51, 14},
    {52, 0},
    {60, 9},
    {70, 15},
    {71, 16},
    {72, 17},
    {80, 18},
    {81, 19},
    {99, 0}};

std::vector<double> label_weight = {0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1.2};

EdgeAnalyticCostFunction::EdgeAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_){

}

bool EdgeAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    
    Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[0] + 4);
    Eigen::Vector3d lp;
    lp = q_last_curr * curr_point + t_last_curr; 

    Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
    Eigen::Vector3d de = last_point_a - last_point_b;
    double de_norm = de.norm();
    residuals[0] = nu.norm()/de_norm;
    
    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Matrix3d skew_lp = skew(lp);
            Eigen::Matrix<double, 3, 6> dp_by_se3;
            dp_by_se3.block<3,3>(0,0) = -skew_lp;
            (dp_by_se3.block<3,3>(0, 3)).setIdentity();
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            Eigen::Matrix3d skew_de = skew(de);
            J_se3.block<1,6>(0,0) = - nu.transpose() / nu.norm() * skew_de * dp_by_se3/de_norm;
      
        }
    }  

    return true;
 
}

LabelEdgeAnalyticCostFunction::LabelEdgeAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_, Eigen::Vector3d last_point_b_,int label_)
    : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_),label(label_)
{
}

bool LabelEdgeAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[0] + 4);
    Eigen::Vector3d lp;
    lp = q_last_curr * curr_point + t_last_curr;

    Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
    Eigen::Vector3d de = last_point_a - last_point_b;
    double de_norm = de.norm();
    // int label_mapped=label_map[label];
    double weight=label_weight[label];


    residuals[0] = (weight)*nu.norm() / de_norm;

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Matrix3d skew_lp = skew(lp);
            Eigen::Matrix<double, 3, 6> dp_by_se3;
            dp_by_se3.block<3, 3>(0, 0) = -skew_lp;
            (dp_by_se3.block<3, 3>(0, 3)).setIdentity();
            // dp_by_se3.block<3, 3>(0, 3)=weight*dp_by_se3.block<3, 3>(0, 3);
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
            J_se3.setZero();
            Eigen::Matrix3d skew_de = skew(de);
            J_se3.block<1, 6>(0, 0) = -nu.transpose() / nu.norm() * skew_de * dp_by_se3 / de_norm;
        }
    }

    return true;
}

SurfNormAnalyticCostFunction::SurfNormAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_) 
                                                        : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_){

}

bool SurfNormAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0] + 4);
    Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;
    residuals[0] = plane_unit_norm.dot(point_w) + negative_OA_dot_norm;

    if(jacobians != NULL)
    {
        if(jacobians[0] != NULL)
        {
            Eigen::Matrix3d skew_point_w = skew(point_w);
            Eigen::Matrix<double, 3, 6> dp_by_se3;
            dp_by_se3.block<3,3>(0,0) = -skew_point_w;
            (dp_by_se3.block<3,3>(0, 3)).setIdentity();
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<1,6>(0,0) = plane_unit_norm.transpose() * dp_by_se3;
   
        }
    }
    return true;

}

LabelSurfNormAnalyticCostFunction::LabelSurfNormAnalyticCostFunction(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_, double negative_OA_dot_norm_,int label_)
    : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), negative_OA_dot_norm(negative_OA_dot_norm_),label(label_)
{
}

bool LabelSurfNormAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0] + 4);
    Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;
    // int label_mapped = label_map[label];
    double weight = label_weight[label];

    residuals[0] =(weight)*( plane_unit_norm.dot(point_w) + negative_OA_dot_norm);

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {
            Eigen::Matrix3d skew_point_w = skew(point_w);
            Eigen::Matrix<double, 3, 6> dp_by_se3;
            dp_by_se3.block<3, 3>(0, 0) = -skew_point_w;
            (dp_by_se3.block<3, 3>(0, 3)).setIdentity();
            // dp_by_se3.block<3, 3>(0, 3) = weight * dp_by_se3.block<3, 3>(0, 3);

            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<1, 6>(0, 0) = plane_unit_norm.transpose() * dp_by_se3;
        }
    }
    return true;
}

bool PoseSE3Parameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> trans(x + 4);

    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_t;
    getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double,6,1>>(delta), delta_q, delta_t);
    Eigen::Map<const Eigen::Quaterniond> quater(x);
    Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);

    quater_plus = delta_q * quater;
    trans_plus = delta_q * trans + delta_t;

    return true;
}

bool PoseSE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    (j.topRows(6)).setIdentity();
    (j.bottomRows(1)).setZero();

    return true;
}

void getTransformFromSe3(const Eigen::Matrix<double,6,1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t){
    Eigen::Vector3d omega(se3.data());
    Eigen::Vector3d upsilon(se3.data()+3);
    Eigen::Matrix3d Omega = skew(omega);

    double theta = omega.norm();
    double half_theta = 0.5*theta;

    double imag_factor;
    double real_factor = cos(half_theta);
    if(theta<1e-10)
    {
        double theta_sq = theta*theta;
        double theta_po4 = theta_sq*theta_sq;
        imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta/theta;
    }

    q = Eigen::Quaterniond(real_factor, imag_factor*omega.x(), imag_factor*omega.y(), imag_factor*omega.z());


    Eigen::Matrix3d J;
    if (theta<1e-10)
    {
        J = q.matrix();
    }
    else
    {
        Eigen::Matrix3d Omega2 = Omega*Omega;
        J = (Eigen::Matrix3d::Identity() + (1-cos(theta))/(theta*theta)*Omega + (theta-sin(theta))/(pow(theta,3))*Omega2);
    }

    t = J*upsilon;
}

Eigen::Matrix<double,3,3> skew(Eigen::Matrix<double,3,1>& mat_in){
    Eigen::Matrix<double,3,3> skew_mat;
    skew_mat.setZero();
    skew_mat(0,1) = -mat_in(2);
    skew_mat(0,2) =  mat_in(1);
    skew_mat(1,2) = -mat_in(0);
    skew_mat(1,0) =  mat_in(2);
    skew_mat(2,0) = -mat_in(1);
    skew_mat(2,1) =  mat_in(0);
    return skew_mat;
}