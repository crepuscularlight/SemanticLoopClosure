#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include <ros/package.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include<cmath>

using namespace std;

string path = ros::package::getPath("iscloam");
Eigen::Matrix3d T1 = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
Eigen::Matrix3d T2 = Eigen::AngleAxisd( M_PI / 2, Eigen::Vector3d(0, 1, 0)).toRotationMatrix();

void odom_callback(const nav_msgs::OdometryConstPtr& msg)
{
    ofstream foutput;

    foutput.open(path+"/trajectory/odom.txt", ios::app);
    double stamp=msg->header.stamp.toSec();
    double tx = msg->pose.pose.position.x;
    double ty = msg->pose.pose.position.y;
    double tz = msg->pose.pose.position.z;
    double qx = msg->pose.pose.orientation.x;
    double qy =msg->pose.pose.orientation.y;
    double qz =msg->pose.pose.orientation.z;
    double qw = msg->pose.pose.orientation.w;

    Eigen::Quaternion<double> q(qw,qx,qy,qz);
    Eigen::Matrix3d r=q.toRotationMatrix();
    Eigen::Vector3d result=(T2*T1).inverse()*Eigen::Vector3d(tx,ty,tz);
    foutput<<r(0,0)<<" "<<r(0,1)<<" "<<r(0,2)<<" "<<result(0)<<" "
    <<r(1,0)<<" "<<r(1,1)<<" "<<r(1,2)<<" "<<result(1)<<" "
    <<r(2,0)<<" "<<r(2,1)<<" "<<r(2,2)<<" "<<result(2)<<
    endl;

    foutput.close();
}

void loop_callback(const nav_msgs::PathConstPtr &msg)
{
    ofstream foutput;

    foutput.open(path + "/trajectory/loop.txt", ios::trunc);

    for(int i=0;i<msg->poses.size();i++)
    {
        geometry_msgs::PoseStamped pose = msg->poses[i];
        double tx = pose.pose.position.x;
        double ty = pose.pose.position.y;
        double tz = pose.pose.position.z;
        double qx = pose.pose.orientation.x;
        double qy = pose.pose.orientation.y;
        double qz = pose.pose.orientation.z;
        double qw = pose.pose.orientation.w;

        Eigen::Quaternion<double> q(qw, qx, qy, qz);
        Eigen::Vector3d result = (T2 * T1).inverse() * Eigen::Vector3d(tx, ty, tz);

        Eigen::Matrix3d r = q.toRotationMatrix();
        foutput << r(0, 0) << " " << r(0, 1) << " " << r(0, 2) << " " << result(0) << " "
                << r(1, 0) << " " << r(1, 1) << " " << r(1, 2) << " " << result(1) << " "
                << r(2, 0) << " " << r(2, 1) << " " << r(2, 2)<< " " << result(2) << endl;
    }


    foutput.close();
}

void gt_callback(const nav_msgs::PathConstPtr &msg)
{
    ofstream foutput;

    foutput.open(path + "/trajectory/gt.txt", ios::trunc);

    for (int i = 0; i < msg->poses.size(); i++)
    {
        geometry_msgs::PoseStamped pose = msg->poses[i];
        double tx = pose.pose.position.x;
        double ty = pose.pose.position.y;
        double tz = pose.pose.position.z;
        double qx = pose.pose.orientation.x;
        double qy = pose.pose.orientation.y;
        double qz = pose.pose.orientation.z;
        double qw = pose.pose.orientation.w;

        Eigen::Quaternion<double> q(qw, qx, qy, qz);
        Eigen::Matrix3d r = q.toRotationMatrix();
        foutput << r(0, 0) << " " << r(0, 1) << " " << r(0, 2) << " " << tx << " "
                << r(1, 0) << " " << r(1, 1) << " " << r(1, 2) << " " << ty << " "
                << r(2, 0) << " " << r(2, 1) << " " << r(2, 2) << " " << tz << endl;
    }

    foutput.close();
}

int main(int argc,char** argv)
{
    ros::init(argc,argv,"Output2txt");
    ros::NodeHandle nh;

    ros::Subscriber sub_odom=nh.subscribe<nav_msgs::Odometry>("/odom_final",100,odom_callback);
    ros::Subscriber sub_loop=nh.subscribe<nav_msgs::Path>("/final_path",100,loop_callback);
    // ros::Subscriber sub_gt=nh.subscribe<nav_msgs::Path>("/gt/trajectory",100,gt_callback);
    ros::spin();
}