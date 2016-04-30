#include <iostream>
#include "vqa_system.hpp"
#include "convert_data.hpp"

int main()
{
//    convert_lmdb();
//    convert_lmdb("question_vector.csv","VAQTrain2014/","answers.csv",
//                 "key_mapping.txt","test.lmdb", ConvertMode::WRITE);

//    convert_lmdb("sample_question.csv", "VAQTrain2014/", "sample_answers.csv", "key_mapping_sample.txt", "sample_vqa_lmdb", ConvertMode::WRITE);
    VQASystem vqas("googlenet_deploy.prototxt", "googlenet.caffemodel",
                   "softmax_deploy.prototxt", "VQA_train_iter_300000.caffemodel");
    
    cv::Mat img = cv::imread("demo.jpg");
    std::vector<std::string> res(vqas.getTopNAnswer(img,
                                                    "What is the weather like today?", 3));
    
    
    std::copy(res.begin(), res.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    
}
