#ifndef feature_extrator_hpp
#define feature_extrator_hpp

#include "data_transformer.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

const int alex_size = 227;
const int googlenet_size = 224;

const int alex_vector_length = 4096;
const int google_vector_length = 1024;


const double red_channel_mean   = 104.00698793;
const double green_channel_mean = 116.66876762;
const double blue_channel_mean  = 122.67891434;

class FeatureExtractor
{
private:
    
    bool is_overlapped(cv::Point point, cv::Rect roi, double stride);
    void modify_proto_file(const std::string& in, const std::string& out,
                           const int rows, const int cols);
    cv::Mat pre_processing(const cv::Mat &image,
                           const std::vector<double> rgb_mean);
    
    std::shared_ptr<caffe::Net<double>> net;
    std::vector<double> _mean;
    
    std::string proto;
    std::string caffe_model;
    
public:
    
    explicit FeatureExtractor(const std::string &proto,
                              const std::string &caffe_model);
    std::vector<double> extract(const cv::Mat &image);

};

#endif /* feature_extrator_hpp */
