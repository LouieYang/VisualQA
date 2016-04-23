#include "vqa_system.hpp"

VQASystem::VQASystem(const string &img_proto, const string &img_model,
                     const string &softmax_proto,
                     const string &softmax_model): fe(img_proto, img_model)
{
    readLabel("key_mapping.txt");
    
    softmax_net.reset(new caffe::Net<double>(softmax_proto,
                                             caffe::Phase::TEST));
    softmax_net->CopyTrainedLayersFrom(softmax_model);
}

void VQASystem::readLabel(const string &addre)
{
    std::ifstream fin(addre, std::ios::in);
    if (!fin.is_open())
        throw std::invalid_argument("Not found key list");
    
    string str;
    while (!fin.eof())
    {
        std::getline(fin, str);
        label.push_back(str.substr(0, str.find_first_of(' ')));
    }
}

vector<double> VQASystem::convertVerbalFeature(const string &addre)
{
    vector<double> vf;
    
    std::ifstream fin(addre, std::ios::in);
    if (!fin.is_open())
        throw std::invalid_argument("Not found verbal feature");
    
    string str;
    std::getline(fin, str);
    
    while (!fin.eof())
    {
        std::getline(fin, str);
        str = str.substr(str.find_last_of(',') + 1);
        if (str.empty())
            break;
        vf.emplace_back(std::stod(str));
    }
    return vf;
}

vector<string> VQASystem::getTopNAnswer(const cv::Mat &img, const string &vf,
                                        int N)
{
    vector<double> img_feature(fe.extract(img));
    
    vector<double> verbal_feature(convertVerbalFeature(vf));
    std::move(img_feature.begin(), img_feature.end(),
              std::back_inserter(verbal_feature));
    
    caffe::Blob<double> *input_layer = softmax_net->input_blobs()[0];
    double *input_data = input_layer->mutable_cpu_data();
    for (auto i = 0; i < verbal_feature.size(); ++i)
    {
        *(input_data + i) = verbal_feature[i];
    }
    softmax_net->ForwardPrefilled();
    
    caffe::Blob<double> *output_layer = softmax_net->output_blobs()[0];
    double *data = const_cast<double*>(output_layer->cpu_data());
    vector<double> scores(data, data + 1000);
    
    vector<int> rank(Argmax(scores, N));
    vector<string> res;
    
    for (int i = 0; i < N; ++i)
    {
        res.push_back(label[rank[i]]);
    }
    return res;
}

static vector<int> Argmax(const std::vector<double> &v, int N)
{
    vector<std::pair<double, int>> pairs;
    for (auto i = 0; i < v.size(); ++i)
    {
        pairs.push_back(std::make_pair(v[i], i));
    }
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(),
                      [](const std::pair<double, int> &lhs,
                         const std::pair<double, int> &rhs)
                      {
                          return lhs.first > rhs.first;
                      });
    
    vector<int> res;
    for (int i = 0; i < N; ++i)
        res.push_back(pairs[i].second);
    return res;
}