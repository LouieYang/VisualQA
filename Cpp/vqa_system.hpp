#ifndef vqa_system_hpp
#define vqa_system_hpp

#include <string>
#include <fstream>
#include <exception>
#include <algorithm>
#include <iterator>
#include <vector>
#include "feature_extractor.hpp"

using std::string;
using std::vector;

class VQASystem
{
    
private:
    FeatureExtractor fe;
    vector<string> label;
    
    std::shared_ptr<caffe::Net<double>> softmax_net;
    
    void readLabel(const string &addre);
    vector<double> convertVerbalFeature(const string &addre);
    
public:
    
    explicit VQASystem(const string &img_proto,
                       const string &img_model,
                       const string &softmax_proto,
                       const string &softmax_model);
    
    vector<string> getTopNAnswer(const cv::Mat &img,
                                 const string &vf, int N);
};

static std::vector<int> Argmax(const std::vector<double> &v,
                               int N);
#endif /* vqa_system_hpp */
