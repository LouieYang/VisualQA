#include "convert_data.hpp"


DEFINE_string(backend, "lmdb",
              "The backend {lmdb, leveldb} for storing the result");

std::condition_variable condition;
std::mutex m;
std::vector<bool> flags;

void read_verbal_feature(std::vector<iidq_feature>& vf,
                         const std::string& file_name)
{
    std::ifstream fin(file_name, std::ios::in);
    
    std::string cell;
    std::string line;
    std::getline(fin, line);
    
    
    int img_id, pre_img_id, ques_id, pre_ques_id;
    double tmp;
    std::getline(fin, line);
    
    std::stringstream lineStream(line);
    std::getline(lineStream, cell, ',');
    pre_img_id = std::stoi(cell);
    std::getline(lineStream, cell, ',');
    pre_ques_id = std::stoi(cell);
    std::getline(lineStream, cell, ',');
    tmp = std::stod(cell);
    
    std::vector<double> tmp_data;
    tmp_data.push_back(tmp);
    while(!fin.eof())
    {
        std::getline(fin, line);
        if (line.empty())
        {
            break;
        }
        std::stringstream lineStream(line);
        std::getline(lineStream, cell, ',');
        img_id = std::stoi(cell);
        std::getline(lineStream, cell, ',');
        ques_id = std::stoi(cell);
        std::getline(lineStream, cell, ',');
        tmp = std::stod(cell);
        
        if(img_id != pre_img_id || ques_id != pre_ques_id)
        {
            vf.emplace_back(make_tuple(pre_img_id,
                                       pre_ques_id, tmp_data));
            tmp_data.clear();
            pre_img_id = img_id;
            pre_ques_id = ques_id;
        }
        tmp_data.push_back(tmp);
    }
    vf.emplace_back(std::make_tuple(pre_img_id, pre_ques_id, tmp_data));
}

void read_verbal_label(std::vector<iida_label>& vl,
                       const std::string& file_name)
{
    using ans_conf = std::pair<std::string, float>;

    auto calculate_word = [](const std::string& sentences)
    {
        return std::distance(std::istream_iterator<std::string>(
                                std::istringstream(sentences) >> std::ws),
                                std::istream_iterator<std::string>());
    };
    
    std::map<std::string, float> confident;
    confident.insert(std::make_pair("yes", 1));
    confident.insert(std::make_pair("no", 0.1));
    confident.insert(std::make_pair("maybe", 0.5));
    
    std::ifstream fin(file_name, std::ios::in);
    
    std::string str;
    std::getline(fin, str);
    
    std::vector<std::string> letters;
    
    int img_id, pre_img_id, ques_id, pre_ques_id;
    std::getline(fin, str);
    split(letters, str, ',');
    CHECK(!letters.empty());
    pre_img_id = std::stod(letters[3]);
    pre_ques_id = std::stod(letters[4]);
    
    std::vector<ans_conf> ans_conf_v;
    ans_conf_v.emplace_back(std::make_pair(letters[1],
                                                 confident[letters[2]]));
    
    while (!fin.eof())
    {
        std::getline(fin, str);
        if (str.empty())
        {
            break;
        }
        split(letters, str, ',');
        img_id = std::stod(letters[3]);
        ques_id = std::stod(letters[4]);
        
        if (img_id == pre_img_id && ques_id == pre_ques_id)
        {
            auto iter = std::find_if(ans_conf_v.begin(),
                                     ans_conf_v.end(),
                                     [letters](ans_conf& value)
                                     {
                                         return value.first == letters[1];
                                     });
            if (iter != ans_conf_v.end())
            {
                iter->second += confident[letters[2]];
                continue;
            }
        }
        else
        {
            auto iter = std::max_element(ans_conf_v.begin(),
                                         ans_conf_v.end(),
                                         [](const ans_conf& a,
                                            const ans_conf& b)
                                         {
                                             return a.second < b.second;
                                         });
            /* If the confident is greater than some threshold */
            if (iter->second > 3 && calculate_word(iter->first) <= 3)
            {
                vl.emplace_back(std::make_tuple(pre_img_id, pre_ques_id,
                                            iter->first));
            }
            
            pre_img_id = img_id;
            pre_ques_id = ques_id;
            
            ans_conf_v.clear();
        }
        ans_conf_v.emplace_back(std::make_pair(letters[1],
                                               confident[letters[2]]));
    }
    auto iter = std::max_element(ans_conf_v.begin(),
                                 ans_conf_v.end(),
                                 [](const ans_conf& a,
                                    const ans_conf& b)
                                 {
                                     return a.second < b.second;
                                 });
    if (iter->second > 3 && calculate_word(iter->first) <= 3)
    {
        vl.emplace_back(std::make_tuple(pre_img_id, pre_ques_id,
                                        iter->first));
    }
}

topLabel convert_label_topN(std::vector<iida_label>& vl,
                            const std::string& key_projection_file,
                            int N, ConvertMode mode)
{
    std::vector<std::pair<std::string, int>> top_N_label;
    if (mode == ConvertMode::WRITE)
    {
        std::vector<std::pair<std::string, int>> label_count;
        for (const auto &l: vl)
        {
            auto iter = std::find_if(label_count.begin(),
                                     label_count.end(),
                                     [&l](const std::pair<std::string, int> a)
                                     {
                                         return a.first == std::get<2>(l);
                                     });
            if (iter == label_count.end())
            {
                label_count.emplace_back(std::make_pair(std::get<2>(l), 1));
            }
            else
            {
                ++(iter->second);
            }
        }
        
        auto pair_compare = [](const std::pair<std::string, int>& lhs,
                               const std::pair<std::string, int>& rhs)
        {
            return lhs.second > rhs.second;
        };
        std::partial_sort(label_count.begin(), label_count.begin() + N,
                          label_count.end(), pair_compare);
        std::move(label_count.begin(), label_count.begin() + N,
                  std::back_inserter(top_N_label));
        
        std::ofstream fout(key_projection_file, std::ios::out);
        unsigned int i = 0;
        for (; i < N - 1; ++i)
        {
            fout << std::get<0>(top_N_label[i]) << " " << i << '\n';
        }
        fout << std::get<0>(top_N_label[i]) << " " << i;
    }
    else if (mode == ConvertMode::READ)
    {
        std::ifstream fin(key_projection_file, std::ios::in);
        for (int i = 0; i < N; ++i)
        {
            std::string str;
            int index;
            fin >> str >> index;
            top_N_label.emplace_back(std::make_pair(str, index));
        }
    }
    return top_N_label;
}

void parallel_merge_feature(unsigned int start, unsigned int end,
                            std::vector<iidq_feature>* verbal_feature,
                            const std::string image_file_in, int thread_index,
                            std::string type)
{
    std::shared_ptr<FeatureExtractor> fe(new FeatureExtractor("googlenet_deploy.prototxt", "googlenet.caffemodel"));
    for (auto i = start; i < end; i = i + 3)
    {
        char index[20];
        std::sprintf(index, "%012d", std::get<0>((*verbal_feature)[i]));
        
        std::string image_address = image_file_in;
        image_address = image_address + type
        + index + ".jpg";
        cv::Mat img = cv::imread(image_address);
        std::vector<double> feature{fe->extract(img)};
        std::copy(feature.begin(), feature.end(),
                  std::back_inserter(std::get<2>((*verbal_feature)[i])));
        std::copy(feature.begin(), feature.end(),
                  std::back_inserter(std::get<2>((*verbal_feature)[i + 1])));
        std::move(feature.begin(), feature.end(),
                  std::back_inserter(std::get<2>((*verbal_feature)[i + 2])));
    }
    flags[thread_index] = true;
    condition.notify_one();
}

void merge_feature(std::vector<iidq_feature> &verbal_feature,
                   const std::string &image_file_in, std::string type)
{
    flags.clear();
    unsigned int max_threads = std::thread::hardware_concurrency() - 1;
    std::vector<std::future<void>> concurrent_thread;
    for (int i = 0; i < max_threads; ++i)
    {
        flags.push_back(false);
    }
    
    unsigned int seg = (verbal_feature.size() / max_threads / 3) * 3;
    unsigned int start = 0, end = seg;
    for (auto i = 0; i < max_threads - 1; ++i)
    {
        concurrent_thread.emplace_back(std::async(std::launch::async,
                                                  parallel_merge_feature,
                                                  start, end, &verbal_feature,
                                                  image_file_in, i, type));
        start = end;
        end += seg;
    }
    concurrent_thread.emplace_back(std::async(std::launch::async,
                                              parallel_merge_feature,
                                              start, verbal_feature.size(),
                                              &verbal_feature, image_file_in,
                                              max_threads - 1, type));
    
    std::unique_lock<std::mutex> lk(m);
    condition.wait(lk, [] {
        for (auto flag: flags)
        {
            if (!flag) return false;
        }
        return true;
    });
}


void convert_lmdb()
{
    std::vector<iidq_feature> verbal_feature_train;
    std::vector<iidq_feature> verbal_feature_val;
    
    read_verbal_feature(verbal_feature_train, "question_vector.csv");
    read_verbal_feature(verbal_feature_val, "question_vector_val.csv");
    
    std::vector<iida_label> train_labels;
    std::vector<iida_label> val_labels;
    
    read_verbal_label(train_labels, "answers.csv");
    read_verbal_label(val_labels, "answers_val.csv");
    
    std::vector<iida_label> merge_labels(train_labels);
    std::copy(val_labels.begin(), val_labels.end(), std::back_inserter(merge_labels));
    topLabel top_N_label(convert_label_topN(merge_labels, "key_mapping.txt", 1000, ConvertMode::WRITE));
    
    std::vector<iida_label>().swap(merge_labels);

    std::sort(verbal_feature_train.begin(), verbal_feature_train.end(),
              [](const iidq_feature& lhs,
                 const iidq_feature& rhs)
              {
                  return std::get<0>(lhs) < std::get<0>(rhs);
              });
    merge_feature(verbal_feature_train, "VQATrain2014/", "COCO_train2014_");
    std::sort(verbal_feature_val.begin(), verbal_feature_val.end(),
              [](const iidq_feature& lhs,
                 const iidq_feature& rhs)
              {
                  return std::get<0>(lhs) < std::get<0>(rhs);
              });
    merge_feature(verbal_feature_val, "val2014/", "COCO_val2014_");
    
    /* Create new DB */
    boost::scoped_ptr<caffe::db::DB> db_train(caffe::db::GetDB(fLS::FLAGS_backend));
    boost::scoped_ptr<caffe::db::DB> db_val(caffe::db::GetDB(fLS::FLAGS_backend));
    db_train->Open("VQA_train_lmdb", caffe::db::NEW);
    db_val->Open("vqa_val_lmdb", caffe::db::NEW);
    boost::scoped_ptr<caffe::db::Transaction> txn_train(db_train->NewTransaction());
    boost::scoped_ptr<caffe::db::Transaction> txn_val(db_val->NewTransaction());
    
    int count = 0;
    for (unsigned int i = 0; i < verbal_feature_train.size(); ++i)
    {
        int ques_id = std::get<1>(verbal_feature_train[i]);
        auto iter_label = std::find_if(train_labels.begin(), train_labels.end(),
                                       [ques_id](iida_label l)
                                       {
                                           return std::get<1>(l) == ques_id;
                                       });
        if (iter_label == train_labels.end())
            continue;
        
        auto iter = std::find_if(top_N_label.begin(), top_N_label.end(),
                                 [iter_label](std::pair<std::string, int> &p)
                                 {
                                     return p.first == std::get<2>(*iter_label);
                                 });
        
        if (iter == top_N_label.end())
            continue;
        std::string out(write_lmdb(std::get<2>(verbal_feature_train[i]),
                                   std::distance(top_N_label.begin(), iter)));
        std::stringstream s;
        s << "train" << std::get<1>(verbal_feature_train[i]);
        txn_train->Put(s.str(), out);
        
        if (++count % 1000 == 0) {
            // Commit db
            txn_train->Commit();
            txn_train.reset(db_train->NewTransaction());
            LOG(ERROR) << "Processed " << count << " vectors.";
        }
    }
    int count2 = 0;
    for (unsigned int i = 0; i < verbal_feature_val.size(); ++i)
    {
        int ques_id = std::get<1>(verbal_feature_val[i]);
        auto iter_label = std::find_if(val_labels.begin(), val_labels.end(),
                                       [ques_id](iida_label l)
                                       {
                                           return std::get<1>(l) == ques_id;
                                       });
        if (iter_label == val_labels.end())
            continue;
        
        auto iter = std::find_if(top_N_label.begin(), top_N_label.end(),
                                 [iter_label](std::pair<std::string, int> &p)
                                 {
                                     return p.first == std::get<2>(*iter_label);
                                 });
        if (iter == top_N_label.end())
            continue;
        std::string out(write_lmdb(std::get<2>(verbal_feature_val[i]),
                                   std::distance(top_N_label.begin(), iter)));
        std::stringstream s;
        s << "val" << std::get<1>(verbal_feature_val[i]);
        
        if (i < int(verbal_feature_val.size() * 0.7))
        {
            txn_train->Put(s.str(), out);
            if (++count % 1000 == 0)
            {
                txn_train->Commit();
                txn_train.reset(db_train->NewTransaction());
                LOG(ERROR) << "Processed " << count << " vectors.";
            }
        }
        else
        {
            txn_val->Put(s.str(), out);
            if (++count2 % 1000 == 0)
            {
                txn_val->Commit();
                txn_val.reset(db_val->NewTransaction());
                LOG(ERROR) << "Processed " << count2 << " vectors.";
            }
        }
    }
    if (count % 1000 != 0) {
        txn_train->Commit();
        LOG(ERROR) << "Processed " << count << " vectors.";
    }
    if (count2 % 1000 != 0) {
        txn_val->Commit();
        LOG(ERROR) << "Processed " << count2 << " vectors.";
    }
}

std::string write_lmdb(std::vector<double> &feature_data, int label)
{
    caffe::Datum datum;
    
    size_t feature_size = feature_data.size();
    for (int j = 0; j < feature_size; ++j)
    {
        datum.add_float_data(feature_data[j]);
    }
    datum.set_channels(int(feature_size));
    datum.set_height(1);
    datum.set_width(1);
    datum.set_label(label);
    
    std::string res;
    CHECK(datum.SerializeToString(&res));
    
    return res;
}