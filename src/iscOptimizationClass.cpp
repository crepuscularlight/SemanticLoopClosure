// Author of ISCLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "iscOptimizationClass.h"
std::map<int, int> label_map_optimize = {
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

ISCOptimizationClass::ISCOptimizationClass()
{
    
}

void ISCOptimizationClass::init(void){

    pointcloud_arr.clear();

    // A prior factor consists of a mean value and a noise model (covariance matrix)
    priorModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6).finished());
    
    // odometry measurement noise model (covariance matrix)
    odomModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) <<  0.10, 0.10, 0.10, 0.10, 0.10, 0.10).finished());

    //loop noise model
    loopModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.10, 0.10, 0.10, 0.10, 0.10, 0.10).finished());
    //attention, here is x y z r p y order
    stop_check_loop_count=0;
    
    downSizeFilter.setLeafSize(0.8, 0.8, 0.8);

}

bool ISCOptimizationClass::addPoseToGraph(const pcl::PointCloud<pcl::PointXYZL>::Ptr& pointcloud_edge_in, const pcl::PointCloud<pcl::PointXYZL>::Ptr& pointcloud_surf_in, std::vector<int>& matched_frame_id, Eigen::Isometry3d& odom_in){

    //pointcloud_arr.push_back(pointcloud_in);

    pcl::VoxelGrid<pcl::PointXYZL> downSizeEdgeFilter;
    pcl::VoxelGrid<pcl::PointXYZL> downSizeSurfFilter;
    downSizeSurfFilter.setLeafSize(0.8, 0.8, 0.8);
    downSizeEdgeFilter.setLeafSize(0.4, 0.4, 0.4);
    // downSizeSurfFilter.setLeafSize(0.4, 0.4, 0.4);
    // downSizeEdgeFilter.setLeafSize(0.4, 0.4, 0.4);
    downSizeEdgeFilter.setInputCloud(pointcloud_edge_in);
    downSizeSurfFilter.setInputCloud(pointcloud_surf_in);
    pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_edge_in(new pcl::PointCloud<pcl::PointXYZL>()); 
    pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_surf_in(new pcl::PointCloud<pcl::PointXYZL>());

    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_edge_in;
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_surf_in;
    for (int i = 0; i < 20; i++)
    {
        dict_edge_in[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_surf_in[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
    }

    downSizeEdgeFilter.filter(*filtered_edge_in);
    downSizeSurfFilter.filter(*filtered_surf_in);
    pointcloud_edge_arr.push_back(filtered_edge_in);
    pointcloud_surf_arr.push_back(filtered_surf_in);

    for (int i = 0; i < filtered_edge_in->points.size(); i++)
    {
        pcl::PointXYZL point_temp;
        point_temp.x = filtered_edge_in->points[i].x;
        point_temp.y = filtered_edge_in->points[i].y;
        point_temp.z = filtered_edge_in->points[i].z;
        point_temp.label = label_map_optimize[filtered_edge_in->points[i].label];

        dict_edge_in[(int)point_temp.label]
            ->push_back(point_temp);

    }

    for (int i = 0; i < filtered_surf_in->points.size(); i++)
    {
        pcl::PointXYZL point_temp;
        point_temp.x = filtered_surf_in->points[i].x;
        point_temp.y = filtered_surf_in->points[i].y;
        point_temp.z = filtered_surf_in->points[i].z;
        point_temp.label = label_map_optimize[filtered_surf_in->points[i].label];

        dict_surf_in[(int)point_temp.label]
            ->push_back(point_temp);
    }
    label_pointcloud_edge_arr.push_back(dict_edge_in);
    label_pointcloud_surf_arr.push_back(dict_surf_in);


    //ROS_INFO("input pc size %d %d",(int)filtered_edge_in->points.size(),(int)filtered_surf_in->points.size());
    gtsam::Pose3 pose3_current = eigenToPose3(odom_in);

    if(pointcloud_edge_arr.size()<=1){
        //if first time 
        pose_optimized_arr.push_back(pose3_current);
        graph.push_back(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', 1), pose_optimized_arr.front(), priorModel));   
        initials.insert(gtsam::Symbol('x', 1), pose_optimized_arr.back());
        last_pose3 = pose3_current;
        return false;
    }

    odom_original_arr.push_back(last_pose3.between(pose3_current));
    pose_optimized_arr.push_back(pose_optimized_arr.back() * odom_original_arr.back());
    last_pose3 = pose3_current;
    initials.insert(gtsam::Symbol('x', pointcloud_edge_arr.size()), pose_optimized_arr.back());
    graph.push_back(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', pointcloud_edge_arr.size()-1), gtsam::Symbol('x', pointcloud_edge_arr.size()), odom_original_arr.back(), odomModel));

    if(stop_check_loop_count>0){
        stop_check_loop_count--;
        return false;
    }
    int current_id = pose_optimized_arr.size()-1;

    if (matched_frame_id.size() != 0)
    {
        //if loop closure detected
        for(int i=0;i<(int)matched_frame_id.size();i++){
            //get initial guess
            // ROS_WARN("matched frame id %d", (int)matched_frame_id.size());
            gtsam::Pose3 transform_pose3 =  pose_optimized_arr[matched_frame_id[i]].between(pose_optimized_arr.back());


            // ROS_WARN("pose %f,%f,%f, [%f,%f,%f,%f]",transform_pose3.translation().x(),transform_pose3.translation().y(),transform_pose3.translation().z(),transform_pose3.rotation().toQuaternion().w(),transform_pose3.rotation().toQuaternion().x(),transform_pose3.rotation().toQuaternion().y(),transform_pose3.rotation().toQuaternion().z());
            Eigen::Isometry3d transform1 = pose3ToEigen(pose_optimized_arr[matched_frame_id[i]].between(pose_optimized_arr.back()));
            Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();

            double label_score = label_estimateOdom(label_pointcloud_edge_arr[matched_frame_id[i]],
                                                    label_pointcloud_surf_arr[matched_frame_id[i]], 
                                                    label_pointcloud_edge_arr.back(),
                                                    label_pointcloud_surf_arr.back(),

                                                    transform);
                                                    
            if(label_score>50)
            {
                transform=transform1;
            }

            if (geometryConsistencyVerification(pointcloud_edge_arr.size() - 1, matched_frame_id[i], transform))
            {

                gtsam::Pose3 loop_temp = eigenToPose3(transform);
                // gtsam::LevenbergMarquardtParams params;
                // params.maxIterations=10;

                graph.push_back(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', matched_frame_id[i] + 1), gtsam::Symbol('x', pointcloud_edge_arr.size()), loop_temp, loopModel));
                gtsam::Values result = gtsam::LevenbergMarquardtOptimizer(graph, initials).optimize();
                ROS_WARN("started update");
                if(updateStates(result, matched_frame_id[i], pointcloud_edge_arr.size()-1)){
                    stop_check_loop_count=STOP_LOOP_CHECK_COUNTER;
                    ROS_WARN("global optimization finished%d,%d with tranform %f,%f,%f, [%f,%f,%f,%f]",pointcloud_edge_arr.size()-1, matched_frame_id[i],loop_temp.translation().x(),loop_temp.translation().y(),loop_temp.translation().z(),loop_temp.rotation().toQuaternion().w(),loop_temp.rotation().toQuaternion().x(),loop_temp.rotation().toQuaternion().y(),loop_temp.rotation().toQuaternion().z());
                    
                    return true;
                }else{
                    stop_check_loop_count=2;
                }
            }
            else
            {
                stop_check_loop_count=2;
            }
        }
    }
    return false;
}

Eigen::Isometry3d ISCOptimizationClass::pose3ToEigen(const gtsam::Pose3& pose3){
    Eigen::Isometry3d pose_eigen = Eigen::Isometry3d::Identity();
    gtsam::Quaternion q_temp = pose3.rotation().toQuaternion();
    pose_eigen.rotate(Eigen::Quaterniond(q_temp.w(),q_temp.x(),q_temp.y(),q_temp.z()));
    pose_eigen.pretranslate(Eigen::Vector3d(pose3.translation().x(),pose3.translation().y(),pose3.translation().z()));
    return pose_eigen;

}

gtsam::Pose3 ISCOptimizationClass::eigenToPose3(const Eigen::Isometry3d& pose_eigen){
    Eigen::Quaterniond q(pose_eigen.rotation());
    return gtsam::Pose3(gtsam::Rot3::Quaternion(q.w(), q.x(), q.y(), q.z()), gtsam::Point3(pose_eigen.translation().x(), pose_eigen.translation().y(), pose_eigen.translation().z()));
}

bool ISCOptimizationClass::updateStates(gtsam::Values& result, int matched_id, int current_id){
    //verify states first
    double sum_residual_q = 0.0;
    double sum_residual_t = 0.0;
    int total_counter=0;
    for(int i =current_id-STOP_LOOP_CHECK_COUNTER-10;i<current_id;i++){
        if(i<0) continue;
        total_counter++;
        gtsam::Pose3 pose_temp1= result.at<gtsam::Pose3>(gtsam::Symbol('x',current_id+1));
        gtsam::Pose3 pose_temp2= result.at<gtsam::Pose3>(gtsam::Symbol('x',i+1));
        gtsam::Pose3 tranform1 = pose_temp2.between(pose_temp1);
        gtsam::Pose3 tranform2 = pose_optimized_arr[i].between(pose_optimized_arr[current_id]);
        gtsam::Pose3 tranform = tranform1.between(tranform2);
        sum_residual_t += std::abs(tranform.translation().x())+std::abs(tranform.translation().y())+std::abs(tranform.translation().z());
        sum_residual_q += std::abs(tranform.rotation().toQuaternion().w()-1)+std::abs(tranform.rotation().toQuaternion().x())+std::abs(tranform.rotation().toQuaternion().y())+std::abs(tranform.rotation().toQuaternion().z());
    }
    for(int i =matched_id-STOP_LOOP_CHECK_COUNTER-10;i<matched_id;i++){
        if(i<0) continue;
        total_counter++;
        // ROS_INFO("total counter %d",total_counter);
        gtsam::Pose3 pose_temp1= result.at<gtsam::Pose3>(gtsam::Symbol('x',matched_id+1));
        gtsam::Pose3 pose_temp2= result.at<gtsam::Pose3>(gtsam::Symbol('x',i+1));
        gtsam::Pose3 tranform1 = pose_temp2.between(pose_temp1);
        gtsam::Pose3 tranform2 = pose_optimized_arr[i].between(pose_optimized_arr[matched_id]);
        gtsam::Pose3 tranform = tranform1.between(tranform2);
        sum_residual_t += std::abs(tranform.translation().x())+std::abs(tranform.translation().y())+std::abs(tranform.translation().z());
        sum_residual_q += std::abs(tranform.rotation().toQuaternion().w()-1)+std::abs(tranform.rotation().toQuaternion().x())+std::abs(tranform.rotation().toQuaternion().y())+std::abs(tranform.rotation().toQuaternion().z());
    }
    sum_residual_q = sum_residual_q / total_counter;
    sum_residual_t = sum_residual_t / total_counter;
    //ROS_INFO("optimization discard due to frame unaligned, residual_q:%f, residual_t:%f",sum_residual_q,sum_residual_t);
    
    if(sum_residual_q>0.05|| sum_residual_t>0.5){
        ROS_INFO("optimization discard due to frame unaligned, residual_q:%f, residual_t:%f",sum_residual_q,sum_residual_t);
        //graph.pop_back();
        graph.remove(graph.size()-1);
        return false;
    }
    //update states
    initials.clear();
    for(int i =0;i<(int)result.size();i++){
        pose_optimized_arr[i]= result.at<gtsam::Pose3>(gtsam::Symbol('x',i+1));
        initials.insert(gtsam::Symbol('x', i+1), pose_optimized_arr[i]);
    }
    return true;


}
bool ISCOptimizationClass::geometryConsistencyVerification(int current_id, int matched_id, Eigen::Isometry3d& transform){
    pcl::PointCloud<pcl::PointXYZL>::Ptr map_surf_temp(new pcl::PointCloud<pcl::PointXYZL>());
    pcl::PointCloud<pcl::PointXYZL>::Ptr map_edge_temp(new pcl::PointCloud<pcl::PointXYZL>());  
    pcl::PointCloud<pcl::PointXYZL>::Ptr map_surf(new pcl::PointCloud<pcl::PointXYZL>()); 
    pcl::PointCloud<pcl::PointXYZL>::Ptr map_edge(new pcl::PointCloud<pcl::PointXYZL>()); 

    for(int i = -10; i <=10; i=i+5){
        if(matched_id+i>= current_id || matched_id+i<0)
            continue;
        Eigen::Isometry3d transform_pose = pose3ToEigen(pose_optimized_arr[matched_id+i]);
        pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*pointcloud_surf_arr[matched_id+i], *transformed_temp, transform_pose.cast<float>());
        *map_surf_temp+=*transformed_temp;
        pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp2(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*pointcloud_edge_arr[matched_id+i], *transformed_temp2, transform_pose.cast<float>());
        *map_edge_temp+=*transformed_temp2;
    }

    Eigen::Isometry3d transform_pose = pose3ToEigen(pose_optimized_arr[matched_id]);
    pcl::transformPointCloud(*map_edge_temp, *map_edge, transform_pose.cast<float>().inverse());
    pcl::transformPointCloud(*map_surf_temp, *map_surf, transform_pose.cast<float>().inverse());
    //ROS_INFO("tag31");
    pcl::PointCloud<pcl::PointXYZL>::Ptr current_scan_surf_temp(new pcl::PointCloud<pcl::PointXYZL>()); 
    pcl::PointCloud<pcl::PointXYZL>::Ptr current_scan_edge_temp(new pcl::PointCloud<pcl::PointXYZL>()); 
    pcl::PointCloud<pcl::PointXYZL>::Ptr current_scan_surf(new pcl::PointCloud<pcl::PointXYZL>()); 
    pcl::PointCloud<pcl::PointXYZL>::Ptr current_scan_edge(new pcl::PointCloud<pcl::PointXYZL>()); 
    for(int i = 0; i <=0; i=i+3){
        if(current_id-i<0)
            continue;
        // std::cout<<"i="<<i<<std::endl;
        // std::cout<<"total:"<<pose_optimized_arr.size()<<std::endl;
        // std::cout<<"current:"<<current_id+i<<std::endl;
        Eigen::Isometry3d transform_pose = pose3ToEigen(pose_optimized_arr[current_id+i]);
        // std::cout<<"x:"<<transform_pose.translation().x()<<std::endl;
        pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*pointcloud_surf_arr[current_id-i], *transformed_temp, transform_pose.cast<float>());
        *current_scan_surf_temp+=*transformed_temp;
        pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp2(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*pointcloud_edge_arr[current_id-i], *transformed_temp2, transform_pose.cast<float>());
        *current_scan_edge_temp+=*transformed_temp2;
    }

    transform_pose = pose3ToEigen(pose_optimized_arr[current_id]);
    pcl::transformPointCloud(*current_scan_edge_temp, *current_scan_edge, transform_pose.cast<float>().inverse());
    pcl::transformPointCloud(*current_scan_surf_temp, *current_scan_surf, transform_pose.cast<float>().inverse());

    //this is for visualization only
    loop_candidate_pc->clear();
    loop_map_pc->clear();
    transform_pose = transform;
    transform_pose.translation() = Eigen::Vector3d(0,0,10);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZL>()); 
    pcl::transformPointCloud(*current_scan_surf, *loop_candidate_pc, transform_pose.cast<float>());
    pcl::transformPointCloud(*current_scan_edge, *cloud_temp, transform_pose.cast<float>());
    *loop_candidate_pc+=*cloud_temp;
    *loop_map_pc += *map_surf;
    *loop_map_pc += *map_edge;

    double match_score = estimateOdom(map_edge,map_surf,current_scan_edge,current_scan_surf,transform);
    // ROS_WARN("matched score %f",match_score);

    if(match_score < LOOPCLOSURE_THRESHOLD){
        return true;
    }
    else{
        ROS_INFO("loop rejected due to geometry verification current_id%d matched_id %d, score: %f",current_id,matched_id,match_score);
        return false;
    }
 return false;
}

bool ISCOptimizationClass::label_geometryConsistencyVerification(int current_id, int matched_id, Eigen::Isometry3d &transform)
{
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_map_surf_temp;
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_map_edge_temp;
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_map_surf;
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_map_edge;
    Eigen::Isometry3d transform_pose = pose3ToEigen(pose_optimized_arr[matched_id]);

    for(int i=0;i<20;i++)
    {
        dict_map_surf_temp[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_map_edge_temp[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_map_surf[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_map_edge[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
    }

    for (int i = -10; i <= 10; i = i + 5)
    {
        if (matched_id + i >= current_id || matched_id + i < 0)
            continue;
        Eigen::Isometry3d transform_pose = pose3ToEigen(pose_optimized_arr[matched_id + i]);
     
        for(int label_i=0;label_i<20;label_i++)
        {   
            pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp(new pcl::PointCloud<pcl::PointXYZL>());
            pcl::transformPointCloud(*label_pointcloud_surf_arr[matched_id + i][label_i], *transformed_temp, transform_pose.cast<float>());
            *dict_map_surf_temp[label_i] += *transformed_temp;
            pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp2(new pcl::PointCloud<pcl::PointXYZL>());
            pcl::transformPointCloud(*label_pointcloud_edge_arr[matched_id + i][label_i], *transformed_temp2, transform_pose.cast<float>());
            *dict_map_edge_temp[label_i] += *transformed_temp2;
        }

    }
    for(int label_i=0;label_i<20;label_i++)
    {
    pcl::transformPointCloud(*dict_map_edge_temp[label_i], *dict_map_edge[label_i], transform_pose.cast<float>().inverse());
    pcl::transformPointCloud(*dict_map_surf_temp[label_i], *dict_map_surf[label_i], transform_pose.cast<float>().inverse());
    }
    // ROS_INFO("tag31");
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_current_scan_surf_temp;
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_current_scan_edge_temp;
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_current_scan_surf;
    std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dict_current_scan_edge;
    for (int i = 0; i < 20; i++)
    {
        dict_current_scan_surf_temp[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_current_scan_edge_temp[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_current_scan_surf[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
        dict_current_scan_edge[i] = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
    }

    for (int i = 0; i <= 0; i = i + 3)
    {
        if (current_id - i < 0)
            continue;
        Eigen::Isometry3d transform_pose = pose3ToEigen(pose_optimized_arr[current_id + i]);
        for(int label_i=0;label_i<20;label_i++)
        {
        pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*label_pointcloud_surf_arr[current_id - i][label_i], *transformed_temp, transform_pose.cast<float>());
        *dict_current_scan_surf_temp[label_i] += *transformed_temp;
        pcl::PointCloud<pcl::PointXYZL>::Ptr transformed_temp2(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*label_pointcloud_edge_arr[current_id - i][label_i], *transformed_temp2, transform_pose.cast<float>());
        *dict_current_scan_edge_temp[label_i] += *transformed_temp2;
        }
    }

    transform_pose = pose3ToEigen(pose_optimized_arr[current_id]);
    for(int label_i=0;label_i<20;label_i++)
    {
    pcl::transformPointCloud(*dict_current_scan_edge_temp[label_i], *dict_current_scan_edge[label_i], transform_pose.cast<float>().inverse());
    pcl::transformPointCloud(*dict_current_scan_surf_temp[label_i], *dict_current_scan_surf[label_i], transform_pose.cast<float>().inverse());
    }
    // this is for visualization only
    // loop_candidate_pc->clear();
    // loop_map_pc->clear();
    // transform_pose = transform;
    // transform_pose.translation() = Eigen::Vector3d(0, 0, 10);
    // pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZL>());
    // pcl::transformPointCloud(*current_scan_surf, *loop_candidate_pc, transform_pose.cast<float>());
    // pcl::transformPointCloud(*current_scan_edge, *cloud_temp, transform_pose.cast<float>());
    // *loop_candidate_pc += *cloud_temp;
    // *loop_map_pc += *map_surf;
    // *loop_map_pc += *map_edge;

    double match_score = label_estimateOdom(dict_map_edge, dict_map_surf, dict_current_scan_edge, dict_current_scan_surf, transform);
    ROS_WARN("matched score %f",match_score);

    if (match_score < LOOPCLOSURE_THRESHOLD)
    {
        return true;
    }
    else
    {
        ROS_INFO("loop rejected due to geometry verification current_id%d matched_id %d, score: %f", current_id, matched_id, match_score);
        return false;
    }
    return false;
}


Eigen::Isometry3d ISCOptimizationClass::getLastPose(void){
    return pose3ToEigen(pose_optimized_arr.back());
}

Eigen::Isometry3d ISCOptimizationClass::getPose(int frame_num){

    return pose3ToEigen(pose_optimized_arr[frame_num]);
}

double ISCOptimizationClass::estimateOdom(const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_source_edge, const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_source_surf, const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_target_edge, const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_target_surf, Eigen::Isometry3d& transform){
    Eigen::Quaterniond init_q(transform.rotation());
    Eigen::Vector3d init_t(0,0,0);
    double parameters[7] = {init_q.x(), init_q.y(), init_q.z(), init_q.w(),init_t.x(), init_t.y(), init_t.z()};
    Eigen::Map<Eigen::Quaterniond> q_temp = Eigen::Map<Eigen::Quaterniond>(parameters);
    Eigen::Map<Eigen::Vector3d> t_temp = Eigen::Map<Eigen::Vector3d>(parameters + 4);
    pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr kdtreeCorner = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
    pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr kdtreeSurf = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
    kdtreeCorner->setInputCloud(pc_source_edge);
    kdtreeSurf->setInputCloud(pc_source_surf);
    double total_cost = 300;
    for (int opti_counter = 0; opti_counter < 20; opti_counter++)
    {
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::Problem::Options problem_options;

        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());
 
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() = t_temp;
        transform.linear() = q_temp.toRotationMatrix();
        //add edge cost factor
        int corner_num=0;
        pcl::PointCloud<pcl::PointXYZL>::Ptr tranformed_edge(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*pc_target_edge, *tranformed_edge, transform);
        for (int i = 0; i < (int) tranformed_edge->points.size(); i++)
        {
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeCorner->nearestKSearch(tranformed_edge->points[i], 5, pointSearchInd, pointSearchSqDis);
            if (pointSearchSqDis[4] < 2.0)
            {
                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; j++)
                {
                    Eigen::Vector3d tmp(pc_source_edge->points[pointSearchInd[j]].x,
                                        pc_source_edge->points[pointSearchInd[j]].y,
                                        pc_source_edge->points[pointSearchInd[j]].z);
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
                Eigen::Vector3d curr_point(pc_target_edge->points[i].x, pc_target_edge->points[i].y, pc_target_edge->points[i].z);
                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                { 
                    Eigen::Vector3d point_on_line = center;
                    Eigen::Vector3d point_a, point_b;
                    point_a = 0.1 * unit_direction + point_on_line;
                    point_b = -0.1 * unit_direction + point_on_line;

                    // ceres::CostFunction *cost_function = new LabelEdgeAnalyticCostFunction(curr_point, point_a, point_b, pc_target_edge->points[i].label);
                    ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
                    problem.AddResidualBlock(cost_function, loss_function, parameters);
                    corner_num++;   
                }                           
            }
        }
        if(corner_num<20){
            ROS_INFO("not enough corresponding points");
            return 300.0;
        }
        //add surf cost factor
        int surf_num=0; 
        pcl::PointCloud<pcl::PointXYZL>::Ptr tranformed_surf(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*pc_target_surf, *tranformed_surf, transform);    
        // find correspondence for plane features
        for (int i = 0; i <(int) tranformed_surf->points.size(); ++i)
        {
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtreeSurf->nearestKSearch(tranformed_surf->points[i], 5, pointSearchInd, pointSearchSqDis);
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (pointSearchSqDis[4] < 2.0)
            {
                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = pc_source_surf->points[pointSearchInd[j]].x;
                    matA0(j, 1) = pc_source_surf->points[pointSearchInd[j]].y;
                    matA0(j, 2) = pc_source_surf->points[pointSearchInd[j]].z;
                }
                
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double negative_OA_dot_norm = 1 / norm.norm();
                norm.normalize();
                bool planeValid = true;
                
                for (int j = 0; j < 5; j++)
                {
                    
                    if (fabs(norm(0) * pc_source_surf->points[pointSearchInd[j]].x +
                             norm(1) * pc_source_surf->points[pointSearchInd[j]].y +
                             norm(2) * pc_source_surf->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                Eigen::Vector3d curr_point(pc_target_surf->points[i].x, pc_target_surf->points[i].y, pc_target_surf->points[i].z);
                if (planeValid)
                {
                    // ceres::CostFunction *cost_function = new LabelSurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, pc_target_edge->points[i].label);
                    ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);
                    problem.AddResidualBlock(cost_function, loss_function, parameters);
                    surf_num++;
                }

            }
        }

        if(surf_num<20){
            ROS_INFO("not enough corresponding points");
            return 300.0;
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if(summary.final_cost<total_cost)
            total_cost = summary.final_cost;
    }
    //transform = Eigen::Isometry3d::Identity();
    transform.linear() = q_temp.toRotationMatrix();
    transform.translation() = t_temp;
    return total_cost;

}

double ISCOptimizationClass::label_estimateOdom(std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr> &dict_pc_source_edge,  
std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> &dict_pc_source_surf,  
std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> & dict_pc_target_edge, 
std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> &dict_pc_target_surf, 
Eigen::Isometry3d &transform)
{
    Eigen::Quaterniond init_q(transform.rotation());
    Eigen::Vector3d init_t(0, 0, 0);
    // Eigen::Vector3d init_t(transform.translation().x(), transform.translation().y(), transform.translation().z());
    double parameters[7] = {init_q.x(), init_q.y(), init_q.z(), init_q.w(), init_t.x(), init_t.y(), init_t.z()};
    Eigen::Map<Eigen::Quaterniond> q_temp = Eigen::Map<Eigen::Quaterniond>(parameters);
    Eigen::Map<Eigen::Vector3d> t_temp = Eigen::Map<Eigen::Vector3d>(parameters + 4);

    // pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr kdtreeCorner = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
    // pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr kdtreeSurf = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());

    std::map<int, pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr> EdgeMapDict;
    std::map<int, pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr> SurfMapDict;
    for (int i = 0; i < 20; i++)
    {
        EdgeMapDict[i] = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
        SurfMapDict[i] = pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZL>());
        // std::cout<<"label"<<i<<"size:"<<dict_pc_source_edge[i]->points.size()<<std::endl;
    }

    // kdtreeCorner->setInputCloud(pc_source_edge);
    // kdtreeSurf->setInputCloud(pc_source_surf);

    for (int update_i = 0; update_i < 20; update_i++)
    {
        if (dict_pc_source_edge[update_i]->points.size()<10)
        {
            continue;
        }
        EdgeMapDict[update_i]->setInputCloud(dict_pc_source_edge[update_i]);
        // std::cout<<"update_i_success"<<std::endl;
        
    }

    for (int update_j = 0; update_j < 20; update_j++)
    {
        if (dict_pc_source_surf[update_j]->points.size() < 10)
        {
            continue;
        }
        SurfMapDict[update_j]->setInputCloud(dict_pc_source_surf[update_j]);
        // std::cout << "update_j_success" << std::endl;
    }

    double total_cost = 300;
    for (int opti_counter = 0; opti_counter < 20; opti_counter++)
    {
        // std::cout<<opti_counter<<"opti_counter success"<<std::endl;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::Problem::Options problem_options;

        ceres::Problem problem(problem_options);
        problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() = t_temp;
        transform.linear() = q_temp.toRotationMatrix();
        // add edge cost factor

        int corner_num_total = 0;
        int surf_num_total = 0;

        for (int label_i=0;label_i<20;label_i++)
        {
            int corner_num = 0;
            int surf_num = 0;
            pcl::PointCloud<pcl::PointXYZL>::Ptr tranformed_edge(new pcl::PointCloud<pcl::PointXYZL>());
            pcl::transformPointCloud(*dict_pc_target_edge[label_i], *tranformed_edge, transform);
            if (dict_pc_source_edge[label_i]->points.size() < 10)
            {
                continue;
            }
            if (dict_pc_source_surf[label_i]->points.size() < 10)
            {
                continue;
            }
            // std::cout << "label:" << label_i << tranformed_edge->points.size()<< std::endl;
            for (int i = 0; i < (int)tranformed_edge->points.size(); i++)
            {
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;
                // kdtreeCorner->nearestKSearch(tranformed_edge->points[i], 5, pointSearchInd, pointSearchSqDis);

                EdgeMapDict[label_i]->nearestKSearch(tranformed_edge->points[i], 5, pointSearchInd, pointSearchSqDis);
                // std::cout << "label:" << label_i << " distance:" << pointSearchSqDis[4] << std::endl;
                if (pointSearchSqDis[4] < 3.0)
                {
                    std::vector<Eigen::Vector3d> nearCorners;
                    Eigen::Vector3d center(0, 0, 0);
                    for (int j = 0; j < 5; j++)
                    {
                        Eigen::Vector3d tmp(dict_pc_source_edge[label_i]->points[pointSearchInd[j]].x,
                                            dict_pc_source_edge[label_i]->points[pointSearchInd[j]].y,
                                            dict_pc_source_edge[label_i]->points[pointSearchInd[j]].z);
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
                    Eigen::Vector3d curr_point(dict_pc_target_edge[label_i]->points[i].x, dict_pc_target_edge[label_i]->points[i].y, dict_pc_target_edge[label_i]->points[i].z);
                    if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                    {
                        Eigen::Vector3d point_on_line = center;
                        Eigen::Vector3d point_a, point_b;
                        point_a = 0.1 * unit_direction + point_on_line;
                        point_b = -0.1 * unit_direction + point_on_line;

                        ceres::CostFunction *cost_function = new LabelEdgeAnalyticCostFunction(curr_point, point_a, point_b, label_i);
                        // ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
                        problem.AddResidualBlock(cost_function, loss_function, parameters);
                        corner_num++;
                        // std::cout<<"corner_num:"<<corner_num<<std::endl;
                    }
                }
            }
            
            corner_num_total+=corner_num;
            // if (corner_num < 20)
            // {
            //     std::cout<<"corner num"<<corner_num<<std::endl;
            //     ROS_INFO("not enough corresponding points");
            //     return 300.0;
            // }
        
        // add surf cost factor
            
            pcl::PointCloud<pcl::PointXYZL>::Ptr tranformed_surf(new pcl::PointCloud<pcl::PointXYZL>());
            pcl::transformPointCloud(*dict_pc_target_surf[label_i], *tranformed_surf, transform);
            // find correspondence for plane features
            for (int i = 0; i < (int)tranformed_surf->points.size(); ++i)
            {
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;

                // kdtreeSurf->nearestKSearch(tranformed_surf->points[i], 5, pointSearchInd, pointSearchSqDis);
                SurfMapDict[label_i]->nearestKSearch(tranformed_surf->points[i], 5, pointSearchInd, pointSearchSqDis);
                Eigen::Matrix<double, 5, 3> matA0;
                Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
                if (pointSearchSqDis[4] < 3.0)
                {
                    for (int j = 0; j < 5; j++)
                    {
                        matA0(j, 0) = dict_pc_source_surf[label_i]->points[pointSearchInd[j]].x;
                        matA0(j, 1) = dict_pc_source_surf[label_i]->points[pointSearchInd[j]].y;
                        matA0(j, 2) = dict_pc_source_surf[label_i]->points[pointSearchInd[j]].z;
                    }

                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();
                    bool planeValid = true;

                    for (int j = 0; j < 5; j++)
                    {

                        if (fabs(norm(0) * matA0(j, 0) +
                                 norm(1) * matA0(j, 1) +
                                 norm(2) * matA0(j, 2) + negative_OA_dot_norm) > 0.2)
                        {
                            planeValid = false;
                            break;
                        }
                    }

                    Eigen::Vector3d curr_point(dict_pc_target_surf[label_i]->points[i].x, dict_pc_target_surf[label_i]->points[i].y, dict_pc_target_surf[label_i]->points[i].z);
                    if (planeValid)
                    {
                        ceres::CostFunction *cost_function = new LabelSurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm, label_i);
                        // ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);
                        problem.AddResidualBlock(cost_function, loss_function, parameters);
                        surf_num++;
                    }
                }
            }
            surf_num_total+=surf_num;
            // if (surf_num< 20)
            // {
            //     std::cout <<"surf num"<< surf_num+corner_num<< std::endl;
            //     ROS_INFO("not enough corresponding points");
            //     return 300.0;
            // }
            // std::cout << "corner_num_total" << corner_num_total << std::endl;
            // std::cout << "surf_num_total" << surf_num_total << std::endl;
        }
        // std::cout << "surf_num_total_out" << surf_num_total << std::endl;
        // std::cout << "corner_num_total_out" << corner_num_total << std::endl;
        if(corner_num_total<50)
        {
            ROS_INFO("not enough corresponding points");

            return 300.0;
        }
        if (surf_num_total <50)
        {
            ROS_INFO("not enough corresponding points");

            return 300.0;
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = 4;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (summary.final_cost < total_cost)
            total_cost = summary.final_cost;
    }
    // std::cout << "solve success" << std::endl;

    // transform = Eigen::Isometry3d::Identity();
    transform.linear() = q_temp.toRotationMatrix();
    transform.translation() = t_temp;
    return total_cost;
}
