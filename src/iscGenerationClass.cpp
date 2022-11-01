// Author of ISCLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "iscGenerationClass.h"
//#define INTEGER_INTENSITY
std::map<int,int> label_map={
  {  0,0},   
  {1 , 0 },    
  {10, 1 },    
  {11, 2 },    
  {13, 5 },    
  {15, 3 },   
  {16, 5 },   
  {18, 4 },   
  {20, 5 },    
  {30, 6 },   
  {31, 7 },   
  {32, 8 },   
  {40, 9 }, 
  {44, 10},    
  {48, 11},    
  {50, 13},    
  {51, 14},  
  {52, 0 },   
  {60, 9 },   
  {70, 15},   
  {71, 16},  
  {72, 17},    
  {80, 18}, 
  {81, 19},   
  {99, 0 }
};
std::vector<int> order_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 15, 16, 14, 17, 9, 18, 19};

ISCGenerationClass::ISCGenerationClass()
{
    
}

void ISCGenerationClass::init_param(int rings_in, int sectors_in, double max_dis_in){
    rings = rings_in;
    sectors = sectors_in;
    max_dis = max_dis_in;
    ring_step = max_dis/rings;
    sector_step = 2*M_PI/sectors;
    print_param();
    init_color();

    current_point_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
}

void ISCGenerationClass::init_color(void){
    for(int i=0;i<1;i++){//RGB format
        color_projection.push_back(cv::Vec3b(0,i*16,255));
    }
    for(int i=0;i<15;i++){//RGB format
        color_projection.push_back(cv::Vec3b(0,i*16,255));
    }
    for(int i=0;i<16;i++){//RGB format
        color_projection.push_back(cv::Vec3b(0,255,255-i*16));
    }
    for(int i=0;i<32;i++){//RGB format
        color_projection.push_back(cv::Vec3b(i*32,255,0));
    }
    for(int i=0;i<16;i++){//RGB format
        color_projection.push_back(cv::Vec3b(255,255-i*16,0));
    }
    for(int i=0;i<64;i++){//RGB format
        color_projection.push_back(cv::Vec3b(i*4,255,0));
    }
    for(int i=0;i<64;i++){//RGB format
        color_projection.push_back(cv::Vec3b(255,255-i*4,0));
    }
    for(int i=0;i<64;i++){//RGB format
        color_projection.push_back(cv::Vec3b(255,i*4,i*4));
    }
}

void ISCGenerationClass::print_param(){
    std::cout << "The ISC parameters are:"<<rings<<std::endl;
    std::cout << "number of rings:\t"<<rings<<std::endl;
    std::cout << "number of sectors:\t"<<sectors<<std::endl;
    std::cout << "maximum distance:\t"<<max_dis<<std::endl;
}

ISCDescriptor ISCGenerationClass::calculate_isc(const pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud){
    ISCDescriptor isc = cv::Mat::zeros(cv::Size(sectors,rings), CV_8U);

    for(int i=0;i<(int)filtered_pointcloud->points.size();i++){
        // ROS_WARN_ONCE("intensity is %f, if intensity showed here is integer format between 1-255, please uncomment #define INTEGER_INTENSITY in iscGenerationClass.cpp and recompile", (double) filtered_pointcloud->points[i].intensity);
        int label = label_map[filtered_pointcloud->points[i].label];
        if (order_vec[label] > 0) 
        {
            double distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
            if(distance>=max_dis || distance <min_dis)
                continue;
            double angle = M_PI + std::atan2(filtered_pointcloud->points[i].y,filtered_pointcloud->points[i].x);
            int ring_id = (distance-min_dis)/ring_step;
            int sector_id = (angle/sector_step);

            if (ring_id >= rings || ring_id < 0)
                continue;
            if (sector_id >= sectors || sector_id < 0)
                continue;

            if (order_vec[label] > order_vec[isc.at<unsigned char>(ring_id, sector_id)])
                    isc.at<unsigned char>(ring_id, sector_id) = label;
        }
    }

    return isc;

}


ISCDescriptor ISCGenerationClass::getLastISCMONO(void){
    return isc_arr.back();

}

ISCDescriptor ISCGenerationClass::getLastISCRGB(void){
    //ISCDescriptor isc = isc_arr.back();
    ISCDescriptor isc_color = cv::Mat::zeros(cv::Size(sectors,rings), CV_8UC3);
    for (int i = 0;i < isc_arr.back().rows;i++) {
        for (int j = 0;j < isc_arr.back().cols;j++) {
            isc_color.at<cv::Vec3b>(i, j) = color_projection[isc_arr.back().at<unsigned char>(i,j)];

        }
    }
    return isc_color;
}

void ISCGenerationClass::loopDetection(const pcl::PointCloud<pcl::PointXYZL>::Ptr& current_pc, Eigen::Isometry3d& odom){

    // std::cout<<"jump into loop detection"<<std::endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr pc_filtered(new pcl::PointCloud<pcl::PointXYZL>());
    ground_filter(current_pc, pc_filtered);
    ISCDescriptor desc = calculate_isc(pc_filtered);
    Eigen::Vector3d current_t = odom.translation();
    // std::cout<<"x:"<<odom.translation().x()<<" "<<odom.translation().y()<<std::endl;
    //dont change push_back sequence
    if(travel_distance_arr.size()==0){
        travel_distance_arr.push_back(0);
    }else{
        double dis_temp = travel_distance_arr.back()+std::sqrt((pos_arr.back()-current_t).array().square().sum());
        travel_distance_arr.push_back(dis_temp);
        // std::cout<<"dist_temp"<<dis_temp<<std::endl;
    }
    pos_arr.push_back(current_t);
    isc_arr.push_back(desc);
    // pointcloud_arr.push_back(current_pc);


    current_frame_id = pos_arr.size()-1;
    matched_frame_id.clear();
    //search for the near neibourgh pos
    int best_matched_id=0;
    double best_score=0.0;
    for(int i = 0; i< (int)pos_arr.size(); i++){
        double delta_travel_distance = travel_distance_arr.back()- travel_distance_arr[i];
        double pos_distance = std::sqrt((pos_arr[i]-pos_arr.back()).array().square().sum());
        // std::cout<<"before delta travel distance"<<std::endl;
        // std::cout <<"delta_distance:"<< delta_travel_distance << std::endl;
        // std::cout << pos_distance << std::endl;
        if(delta_travel_distance > SKIP_NEIBOUR_DISTANCE && pos_distance<delta_travel_distance*INFLATION_COVARIANCE){
            double geo_score=0;
            double inten_score =0;
            // std::cout<<"before loop pair"<<std::endl;
            if(is_loop_pair(desc,isc_arr[i],
            //current_pc,pointcloud_arr[i],
            geo_score,inten_score)){
                if(geo_score+inten_score>best_score){
                    best_score = geo_score+inten_score;
                    best_matched_id = i;
                }
            }

        }
    }
    if(best_matched_id!=0){
        matched_frame_id.push_back(best_matched_id);
        // ROS_INFO("received loop closure candidate: current: %d, history %d, total_score%f",current_frame_id,best_matched_id,best_score);
    }


}

bool ISCGenerationClass::is_loop_pair(ISCDescriptor &desc1, ISCDescriptor &desc2,
                                      //pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2,
                                       double &geo_score, double &inten_score)
{
    int angle =0;
    // std::cout<<"jump into loop pair"<<std::endl;
    geo_score = calculate_geometry_dis(desc1,desc2,angle);
    if(geo_score>GEOMETRY_THRESHOLD){

        inten_score = calculate_intensity_dis(desc1,desc2,angle);
        // std::cout<<"inten_score"<<inten_score<<std::endl;
        if(inten_score>INTENSITY_THRESHOLD){
            // std::cout << "inten_score" << inten_score << std::endl;
            return true;          
        }
    }
    return false;
}

double ISCGenerationClass::calculate_geometry_dis(const ISCDescriptor& desc1, const ISCDescriptor& desc2, int& angle){
    double similarity = 0.0;

    for(int i=0;i<sectors;i++){
        int match_count=0;
        for(int p=0;p<sectors;p++){
            int new_col = p+i>=sectors?p+i-sectors:p+i;
            for(int q=0;q<rings;q++){
                if((desc1.at<unsigned char>(q,p)== true && desc2.at<unsigned char>(q,new_col)== true) || (desc1.at<unsigned char>(q,p)== false && desc2.at<unsigned char>(q,new_col)== false)){
                    match_count++;
                }

            }
        }
        if(match_count>similarity){
            similarity=match_count;
            angle = i;
        }

    }
    return similarity/(sectors*rings);
    
}
double ISCGenerationClass::calculate_intensity_dis(const ISCDescriptor& desc1, const ISCDescriptor& desc2, int& angle){
    double difference = 1.0;
    double angle_temp = angle;
    for(int i=angle_temp-30;i<angle_temp+30;i++){

        int match_count=0;
        int total_points=0;
        for(int p=0;p<sectors;p++){
            int new_col = p+i;
            if(new_col>=sectors)
                new_col = new_col-sectors;
            if(new_col<0)
                new_col = new_col+sectors;
            for(int q=0;q<rings;q++){
                    match_count += abs(desc1.at<unsigned char>(q,p)-desc2.at<unsigned char>(q,new_col));
                    total_points++;
            }
            
        }
        double diff_temp = ((double)match_count)/(sectors*rings*20);
        if(diff_temp<difference)
            difference=diff_temp;

    }
    return 1 - difference;
    // double similarity = 0;
    // int sectors = desc1.cols;
    // int rings = desc1.rows;
    // int valid_num = 0;
    // for (int p = 0; p < sectors; p++)
    // {
    //     for (int q = 0; q < rings; q++)
    //     {
    //         if (desc1.at<unsigned char>(q, p) == 0 && desc2.at<unsigned char>(q, p) == 0)
    //         {
    //             continue;
    //         }
    //         valid_num++;

    //         if (desc1.at<unsigned char>(q, p) == desc2.at<unsigned char>(q, p))
    //         {
    //             similarity++;
    //         }
    //     }
    // }

    // return similarity / valid_num;
}
void ISCGenerationClass::ground_filter(const pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_in, pcl::PointCloud<pcl::PointXYZL>::Ptr& pc_out){
    pcl::PassThrough<pcl::PointXYZL> pass;
    pass.setInputCloud (pc_in);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-0.9, 30.0);
    pass.filter (*pc_out);

}

cv::Mat ISCGenerationClass::project(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud)
{
    auto sector_step = 2. * M_PI / sectors_range;
    cv::Mat ssc_dis = cv::Mat::zeros(cv::Size(sectors_range, 1), CV_32FC4);
    for (uint i = 0; i < filtered_pointcloud->points.size(); i++)
    {
        auto label = label_map[filtered_pointcloud->points[i].label];
        if (label == 13 || label == 14 || label == 16 || label == 18 || label == 19)
        {
            float distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
            if (distance < 1e-2)
            {
                continue;
            }
            // int sector_id = cv::fastAtan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x);
            float angle = M_PI + std::atan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x);
            int sector_id = std::floor(angle / sector_step);
            if (sector_id >= sectors_range || sector_id < 0)
                continue;
            // if(ssc_dis.at<cv::Vec4f>(0, sector_id)[3]<10||distance<ssc_dis.at<cv::Vec4f>(0, sector_id)[0]){
            ssc_dis.at<cv::Vec4f>(0, sector_id)[0] = distance;
            ssc_dis.at<cv::Vec4f>(0, sector_id)[1] = filtered_pointcloud->points[i].x;
            ssc_dis.at<cv::Vec4f>(0, sector_id)[2] = filtered_pointcloud->points[i].y;
            ssc_dis.at<cv::Vec4f>(0, sector_id)[3] = label;
            // }
        }
    }
    return ssc_dis;
}

Eigen::Matrix4f ISCGenerationClass::globalICP(cv::Mat &ssc_dis1, cv::Mat &ssc_dis2)
{
    double similarity = 100000;
    float angle = 0;
    int sectors = ssc_dis1.cols;
    for (int i = 0; i < sectors; ++i)
    {
        float dis_count = 0;
        for (int j = 0; j < sectors; ++j)
        {
            int new_col = j + i >= sectors ? j + i - sectors : j + i;
            cv::Vec4f vec1 = ssc_dis1.at<cv::Vec4f>(0, j);
            cv::Vec4f vec2 = ssc_dis2.at<cv::Vec4f>(0, new_col);
            // if(vec1[3]==vec2[3]){
            dis_count += fabs(vec1[0] - vec2[0]);
            // }
        }
        if (dis_count < similarity)
        {
            similarity = dis_count;
            angle = i;
        }
    }
    angle = M_PI * (360. - angle * 360. / sectors) / 180.;
    auto cs = cos(angle);
    auto sn = sin(angle);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>), cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    // recover the rotation of point clouds
    for (int i = 0; i < sectors; ++i)
    {
        if (ssc_dis1.at<cv::Vec4f>(0, i)[3] > 0)
        {
            cloud1->push_back(pcl::PointXYZ(ssc_dis1.at<cv::Vec4f>(0, i)[1], ssc_dis1.at<cv::Vec4f>(0, i)[2], 0.));
        }
        if (ssc_dis2.at<cv::Vec4f>(0, i)[3] > 0)
        {
            float tpx = ssc_dis2.at<cv::Vec4f>(0, i)[1] * cs - ssc_dis2.at<cv::Vec4f>(0, i)[2] * sn;
            float tpy = ssc_dis2.at<cv::Vec4f>(0, i)[1] * sn + ssc_dis2.at<cv::Vec4f>(0, i)[2] * cs;
            cloud2->push_back(pcl::PointXYZ(tpx, tpy, 0.));
        }
    }
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    icp.setInputSource(cloud2);
    icp.setInputTarget(cloud1);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    auto trans = icp.getFinalTransformation();
    Eigen::Affine3f trans1 = Eigen::Affine3f::Identity();
    trans1.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
    return trans * trans1.matrix();
}