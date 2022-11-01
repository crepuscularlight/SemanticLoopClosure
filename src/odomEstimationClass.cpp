// Author of ISCLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "odomEstimationClass.h"
std::map<int, int> label_map_odom = {
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

void OdomEstimationClass::init(lidar::Lidar lidar_param, double map_resolution){
    //init local map
    laserCloudCornerMap = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
    laserCloudSurfMap = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());

    for(int i=0;i<20;i++)
    {
        labellaserCloudCornerMap[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        labellaserCloudSurfMap[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
    }
    
    // init map dict
    for(int i=0;i<20;i++)
    {
        EdgeMapDict[i] = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
        SurfMapDict[i] = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
    }

    //downsampling size
    downSizeFilterEdge.setLeafSize(map_resolution, map_resolution, map_resolution);
    downSizeFilterSurf.setLeafSize(map_resolution *2, map_resolution * 2, map_resolution * 2);

    //kd-tree
    kdtreeEdgeMap = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
    kdtreeSurfMap = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());

    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    optimization_count=2;
}

void OdomEstimationClass::initMapWithPoints(const pcl::PointCloud<pcl::PointXYZL>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZL>::Ptr& surf_in){
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;

    for (int i = 0; i < edge_in->points.size(); i++)
    {
        pcl::PointXYZL point_temp;
        point_temp.x = edge_in->points[i].x;
        point_temp.y = edge_in->points[i].y;
        point_temp.z = edge_in->points[i].z;
        point_temp.label = label_map_odom[edge_in->points[i].label];

        // laserCloudCornerMap->points[i].label = label_map_odom[laserCloudCornerMap->points[i].label];
        labellaserCloudCornerMap[(int)point_temp.label]
            ->push_back(point_temp);
        // std::cout << "key label:" << (int)label_map_odom[(int)laserCloudCornerMap->points[i].label] << " point label:"<<point_temp.label<< std::endl;
    }

    for (int i = 0; i < surf_in->points.size(); i++)
    {
        pcl::PointXYZL point_temp1;
        point_temp1.x = surf_in->points[i].x;
        point_temp1.y = surf_in->points[i].y;
        point_temp1.z = surf_in->points[i].z;
        point_temp1.label = label_map_odom[(int)surf_in->points[i].label];

        // laserCloudCornerMap->points[i].label = label_map_odom[laserCloudCornerMap->points[i].label];
        labellaserCloudSurfMap[(int)point_temp1.label]
            ->push_back(point_temp1);
        // std::cout << "key label:" << (int)label_map_odom[(int)laserCloudSurfMap->points[i].label] << " point label:"<<point_temp.label<< std::endl;
    }

    optimization_count=12;
}


void OdomEstimationClass::updatePointsToMap(const pcl::PointCloud<pcl::PointXYZL>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZL>::Ptr& surf_in){

    if(optimization_count>2)
        optimization_count--;

    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    pcl::PointCloud<pcl::PointXYZL>::Ptr downsampledEdgeCloud(new pcl::PointCloud<pcl::PointXYZL>());
    pcl::PointCloud<pcl::PointXYZL>::Ptr label_downsampledEdgeCloud(new pcl::PointCloud<pcl::PointXYZL>());
    pcl::PointCloud<pcl::PointXYZL>::Ptr downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZL>());  
    pcl::PointCloud<pcl::PointXYZL>::Ptr label_downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZL>());
    // downSamplingToMap(edge_in,downsampledEdgeCloud,surf_in,downsampledSurfCloud);
    label_downSamplingToMap(edge_in,label_downsampledEdgeCloud,surf_in,label_downsampledSurfCloud);
    //ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
    if(laserCloudCornerMap->points.size()>10 && laserCloudSurfMap->points.size()>50){
        // kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        // kdtreeSurfMap->setInputCloud(laserCloudSurfMap);

        for(int update_i=0;update_i<20;update_i++)
        {
            if (labellaserCloudCornerMap[update_i]->points.size() == 0)
            {
                continue;
            }
            EdgeMapDict[update_i]->setInputCloud(labellaserCloudCornerMap[update_i]);
        }

        for (int update_j = 0; update_j < 20; update_j++)
        {
            if (labellaserCloudSurfMap[update_j]->points.size() == 0)
            {
                continue;
            }
            SurfMapDict[update_j]->setInputCloud(labellaserCloudSurfMap[update_j]);
        }

        for (int iterCount = 0; iterCount < optimization_count; iterCount++)
        {
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);

            problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

            addEdgeCostFactor(label_downsampledEdgeCloud, labellaserCloudCornerMap, problem, loss_function);
            // addEdgeCostFactor(label_downsampledEdgeCloud, laserCloudCornerMap, problem, loss_function);
            // addSurfCostFactor(label_downsampledSurfCloud, laserCloudSurfMap, problem, loss_function);
            addSurfCostFactor(label_downsampledSurfCloud, labellaserCloudSurfMap, problem, loss_function);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;

            ceres::Solve(options, &problem, &summary);
        }
    }else{
        printf("not enough points in map to associate, map error");
    }
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;
    addPointsToMap(label_downsampledEdgeCloud,label_downsampledSurfCloud);

}

void OdomEstimationClass::pointAssociateToMap(pcl::PointXYZL const *const pi, pcl::PointXYZL *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->label =pi->label;
    //po->intensity = 1.0;
}

void OdomEstimationClass::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZL>::Ptr& edge_pc_in, pcl::PointCloud<pcl::PointXYZL>::Ptr& edge_pc_out, const pcl::PointCloud<pcl::PointXYZL>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZL>::Ptr& surf_pc_out){
    downSizeFilterEdge.setInputCloud(edge_pc_in);
    downSizeFilterEdge.filter(*edge_pc_out);
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);    
}

void OdomEstimationClass::label_downSamplingToMap(const pcl::PointCloud<pcl::PointXYZL>::Ptr& edge_pc_in, pcl::PointCloud<pcl::PointXYZL>::Ptr& edge_pc_out, const pcl::PointCloud<pcl::PointXYZL>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZL>::Ptr& surf_pc_out){
    // downSizeFilterEdge.setInputCloud(edge_pc_in);
    // downSizeFilterEdge.filter(*edge_pc_out);
    // downSizeFilterSurf.setInputCloud(surf_pc_in);
    // downSizeFilterSurf.filter(*surf_pc_out); 
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_edge_in;
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_surf_in;
    for(int i=0;i<20;i++)
    {
        dict_edge_in[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_surf_in[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
    }
    for (int i = 0; i < edge_pc_in->points.size(); i++)
    {
        pcl::PointXYZL point_temp;
        point_temp.x = edge_pc_in->points[i].x;
        point_temp.y = edge_pc_in->points[i].y;
        point_temp.z = edge_pc_in->points[i].z;
        int label_temp = label_map_odom[edge_pc_in->points[i].label];

        // laserCloudCornerMap->points[i].label = label_map_odom[laserCloudCornerMap->points[i].label];
        dict_edge_in[label_temp]
            ->push_back(point_temp);
        // std::cout << "key label:" << (int)label_map_odom[(int)laserCloudCornerMap->points[i].label] << " point label:"<<point_temp.label<< std::endl;
    }

    for (int i = 0; i < surf_pc_in->points.size(); i++)
    {
        pcl::PointXYZL point_temp1;
        point_temp1.x = surf_pc_in->points[i].x;
        point_temp1.y = surf_pc_in->points[i].y;
        point_temp1.z = surf_pc_in->points[i].z;
        int label_tmp = label_map_odom[(int)surf_pc_in->points[i].label];

        // laserCloudCornerMap->points[i].label = label_map_odom[laserCloudCornerMap->points[i].label];
        dict_surf_in[label_tmp]
            ->push_back(point_temp1);
        // std::cout << "key label:" << (int)label_map_odom[(int)laserCloudSurfMap->points[i].label] << " point label:"<<point_temp.label<< std::endl;
    } 

    for (int label_i=0;label_i<20;label_i++)
    {
        
        pcl::PointCloud<pcl::PointXYZL>::Ptr edge_out_tmp(new pcl::PointCloud<pcl::PointXYZL>());
        downSizeFilterEdge.setInputCloud(dict_edge_in[label_i]);
        downSizeFilterEdge.filter(*edge_out_tmp);
        *edge_pc_out+=*edge_out_tmp; 


        pcl::PointCloud<pcl::PointXYZL>::Ptr surf_out_tmp(new pcl::PointCloud<pcl::PointXYZL>());
        downSizeFilterSurf.setInputCloud(dict_surf_in[label_i]);
        downSizeFilterSurf.filter(*surf_out_tmp);
        *surf_pc_out+=*surf_out_tmp; 
    }
}

void OdomEstimationClass::addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZL>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function){
    int corner_num=0;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {

        pcl::PointXYZL point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

        if (pointSearchSqDis[4] < 1)
            {
                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                        map_in->points[pointSearchInd[j]].y,
                                        map_in->points[pointSearchInd[j]].z);
                    center = center + tmp;
                    nearCorners.push_back(tmp);
                    // std::cout << "label origin" << label_map_odom[point_temp.label] << " "
                            //   << "label" << j << " " << label_map_odom[map_in->points[pointSearchInd[j]].label] << std::endl;
                }
                center = center / 5.0;

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                    covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                {
                    Eigen::Vector3d point_on_line = center;
                    Eigen::Vector3d point_a, point_b;
                    point_a = 0.1 * unit_direction + point_on_line;
                    point_b = -0.1 * unit_direction + point_on_line;

                    // ceres::CostFunction *cost_function = new LabelEdgeAnalyticCostFunction(curr_point, point_a, point_b, pc_in->points[i].label);
                    ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
                    problem.AddResidualBlock(cost_function, loss_function, parameters);
                    corner_num++;
                }
            }
    }
    if(corner_num<20){
        std::cout<<corner_num<<std::endl;
        printf("not enough correct points");
    }

}
void OdomEstimationClass::addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZL>::Ptr &pc_in, std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> &dict_map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int corner_num = 0;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {

        pcl::PointXYZL point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        
        int label_temp = label_map_odom[(int)point_temp.label];
        point_temp.label=label_temp;
        if (dict_map_in[label_temp]->points.size()<5)
        {
            continue;
        }

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        EdgeMapDict[(int)point_temp.label]->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        // std::cout <<"distance"<<pointSearchSqDis[4] << std::endl;
        if (pointSearchSqDis[4] < 1)
        {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0);
            
            for (int j = 0; j < 5; j++)
            {
                
                Eigen::Vector3d tmp(dict_map_in[label_temp]->points[pointSearchInd[j]].x,
                                    dict_map_in[label_temp]->points[pointSearchInd[j]].y,
                                    dict_map_in[label_temp]->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);

            }
            center = center / 5.0;

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++)
            {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                ceres::CostFunction *cost_function = new LabelEdgeAnalyticCostFunction(curr_point, point_a, point_b, label_temp);
                // ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
                problem.AddResidualBlock(cost_function, loss_function, parameters);
                corner_num++;
            }
        }
    }
    // std::cout<<"corner num:"<<corner_num<<std::endl;
    if (corner_num < 20)
    {
        std::cout << corner_num << std::endl;
        printf("not enough correct points");
    }
}

void OdomEstimationClass::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZL>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function){
    int surf_num=0;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {

        pcl::PointXYZL point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0)
        {
            
            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            if (planeValid)
            {
                // ceres::CostFunction *cost_function = new LabelSurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, pc_in->points[i].label);
                ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, parameters);

                surf_num++;
            }
        }

    }
    if(surf_num<20){
        printf("not enough correct points");
    }

}

void OdomEstimationClass::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZL>::Ptr &pc_in, std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> &dict_map_in, ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    int surf_num = 0;
   

    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {

        pcl::PointXYZL point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        int label_temp = label_map_odom[(int)point_temp.label];
        point_temp.label = label_temp;

        if (dict_map_in[label_temp]->points.size() < 5)
        {
            continue;
        }
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        SurfMapDict[label_temp]->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        // std::cout << pointSearchSqDis[4] << std::endl;
        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1)
        {

            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = dict_map_in[label_temp]->points[pointSearchInd[j]].x;
                matA0(j, 1) = dict_map_in[label_temp]->points[pointSearchInd[j]].y;
                matA0(j, 2) = dict_map_in[label_temp]->points[pointSearchInd[j]].z;
                // std::cout << "origin label " << label_temp << "current label " << dict_map_in[label_temp]->points[pointSearchInd[j]].label<< std::endl;
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * matA0(j, 0) +
                         norm(1) * matA0(j, 1) +
                         norm(2) * matA0(j, 2) + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            if (planeValid)
            {
                ceres::CostFunction *cost_function = new LabelSurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, label_temp);
                // ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function, parameters);

                surf_num++;
            }
        }
    }
    // std::cout << "surf num:" << surf_num << std::endl;

    if (surf_num < 20)
    {
        std::cout<<"surf_num:"<<surf_num<<std::endl;
        printf("not enough correct points");
    }
}

void OdomEstimationClass::addPointsToMap(const pcl::PointCloud<pcl::PointXYZL>::Ptr& downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZL>::Ptr& downsampledSurfCloud){
    //储存点
    for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++)
    {
        pcl::PointXYZL point_temp;
        pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
        laserCloudCornerMap->push_back(point_temp);
        point_temp.label = label_map_odom[downsampledEdgeCloud->points[i].label];
        labellaserCloudCornerMap[point_temp.label]
                               ->push_back(point_temp);
    }
    
    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
    {
        pcl::PointXYZL point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap->push_back(point_temp);
        point_temp.label = label_map_odom[downsampledSurfCloud->points[i].label];
        labellaserCloudSurfMap[point_temp.label]
            ->push_back(point_temp);
    }
    
    double x_min = +odom.translation().x()-100;
    double y_min = +odom.translation().y()-100;
    double z_min = +odom.translation().z()-100;
    double x_max = +odom.translation().x()+100;
    double y_max = +odom.translation().y()+100;
    double z_max = +odom.translation().z()+100;
    
    //ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);    

    pcl::PointCloud<pcl::PointXYZL>::Ptr tmpCorner(new pcl::PointCloud<pcl::PointXYZL>());
    pcl::PointCloud<pcl::PointXYZL>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZL>());
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);
    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*tmpCorner);
    for (int i=0;i<20;i++)
    {
        pcl::PointCloud<pcl::PointXYZL>::Ptr tmp_dict(new pcl::PointCloud<pcl::PointXYZL>());
        if(labellaserCloudCornerMap[i]->points.size()==0)
        {
            continue;
        }
        cropBoxFilter.setInputCloud(labellaserCloudCornerMap[i]);
        cropBoxFilter.filter(*tmp_dict);

        downSizeFilterEdge.setInputCloud(tmp_dict);
        downSizeFilterEdge.filter(*labellaserCloudCornerMap[i]);
    }

    for (int i = 0; i < 20; i++)
    {
        pcl::PointCloud<pcl::PointXYZL>::Ptr tmp_dict1(new pcl::PointCloud<pcl::PointXYZL>());
        if (labellaserCloudSurfMap[i]->points.size() == 0)
        {
            continue;
        }
        cropBoxFilter.setInputCloud(labellaserCloudSurfMap[i]);
        cropBoxFilter.filter(*tmp_dict1);

        downSizeFilterSurf.setInputCloud(tmp_dict1);
        downSizeFilterSurf.filter(*labellaserCloudSurfMap[i]);
    }

    downSizeFilterSurf.setInputCloud(tmpSurf);
    downSizeFilterSurf.filter(*laserCloudSurfMap);
    downSizeFilterEdge.setInputCloud(tmpCorner);
    downSizeFilterEdge.filter(*laserCloudCornerMap);

}

void OdomEstimationClass::getMap(pcl::PointCloud<pcl::PointXYZL>::Ptr& laserCloudMap){
    
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

OdomEstimationClass::OdomEstimationClass(){

}