#ifndef convert_data_hpp
#define convert_data_hpp

#include <map>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>
#include <regex>
#include <sstream>
#include <istream>
#include <cctype>

#include "feature_extractor.hpp"

#include <thread>
#include <future>
#include <condition_variable>
#include <mutex>

#include <fstream>
#include <utility>
#include <boost/scoped_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>
#include <caffe/util/rng.hpp>

using topLabel = std::vector<std::pair<std::string, int>>;

const int nil = -1;
enum class ConvertMode
{
    READ, WRITE
};

/* img_id, ques_id, label */
using iidq_feature = std::tuple<int, int, std::vector<double>>;
using iida_label = std::tuple<int, int, std::string>;
void read_verbal_feature(std::vector<iidq_feature>& vf,
                         const std::string& file_name);
void read_verbal_label(std::vector<iida_label>& vl,
                       const std::string& file_name);
topLabel convert_label_topN(std::vector<iida_label>& vl,
                        const std::string& key_projection_file,
                        int N, ConvertMode mode);

void convert_lmdb();

void split(std::vector<std::string>& elements, std::string str,
           const std::string& regex);

void split(std::vector<std::string>& elements, std::string str,
           char regex);

void parallel_merge_feature(unsigned int start, unsigned int end,
                            std::vector<iidq_feature>* verbal_feature,
                            const std::string image_file_in,
                            int thread_index, std::string type);
void merge_feature(std::vector<iidq_feature> &verbal_feature,
                   const std::string &image_file_in, std::string type);

std::string write_lmdb(std::vector<double> &feature_data, int label);
#endif /* convert_data_hpp */
