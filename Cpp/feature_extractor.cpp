#include "feature_extractor.hpp"

inline bool FeatureExtractor::is_overlapped(cv::Point point,
                                            cv::Rect roi, double stride)
{
    return !((point.y < roi.y - stride) || (point.x < roi.x - stride)
             || (point.x > roi.x + roi.width + stride) || (point.y > roi.y + roi.height + stride));
}

cv::Mat FeatureExtractor::pre_processing(const cv::Mat &image,
                                    const std::vector<double> rgb_mean)
{
    const double stride = 10;
    
    std::vector<std::pair<cv::Mat, cv::Point>> pyramid;
    double scale = 2;
    const double decay = std::pow(3, 1./8);
    do
    {
        cv::Mat Tmp;
        cv::resize(image, Tmp, cv::Size(image.cols * scale, image.rows * scale));
        scale /= decay;
        pyramid.push_back(std::make_pair(std::move(Tmp), cv::Point(-1, -1)));
    } while (MIN(pyramid[pyramid.size() - 1].first.cols, pyramid[pyramid.size() - 1].first.rows) > 100);
    
    cv::Mat large_plane;
    assert(rgb_mean.size() == image.channels());
    std::vector<cv::Mat> channel(image.channels());
    
    if (image.cols > image.rows)
    {
        for (unsigned int i = 0; i < image.channels(); ++i)
        {
            channel[i] = cv::Mat::ones(pyramid[0].first.cols + pyramid[1].first.cols + 2 * stride,
                                       pyramid[0].first.cols + pyramid[1].first.cols + 2 * stride,
                                       CV_8UC1) * rgb_mean[i];
        }
        cv::merge(channel, large_plane);
    }
    else
    {
        for (unsigned int i = 0; i < image.channels(); ++i)
        {
            channel[i] = cv::Mat::ones(pyramid[0].first.rows + pyramid[1].first.rows + 2 * stride,
                                       pyramid[0].first.rows + pyramid[1].first.rows + 2 * stride,
                                       CV_8UC1) * rgb_mean[i];
        }
        cv::merge(channel, large_plane);
    }
    
    unsigned int index = 0;
    while (index != pyramid.size())
    {
        bool is_copied = false;
        for (unsigned int i = 0; i < large_plane.rows; ++i)
        {
            for (unsigned int j = 0; j < large_plane.cols; ++j)
            {
                unsigned int id = 0;
                for (; id < index; id++)
                {
                    if (pyramid[id].second.x < 0)
                    {
                        continue;
                    }
                    if (is_overlapped(cv::Point(j, i),
                                      cv::Rect(pyramid[id].second.x, pyramid[id].second.y,
                                               pyramid[id].first.cols, pyramid[id].first.rows), stride) ||
                        is_overlapped(cv::Point(j + pyramid[index].first.cols, i),
                                      cv::Rect(pyramid[id].second.x, pyramid[id].second.y,
                                               pyramid[id].first.cols, pyramid[id].first.rows), stride) ||
                        is_overlapped(cv::Point(j, i + pyramid[index].first.rows),
                                      cv::Rect(pyramid[id].second.x, pyramid[id].second.y,
                                               pyramid[id].first.cols, pyramid[id].first.rows), stride) ||
                        is_overlapped(cv::Point(j + pyramid[index].first.cols, i + pyramid[index].first.rows),
                                      cv::Rect(pyramid[id].second.x, pyramid[id].second.y,
                                               pyramid[id].first.cols, pyramid[id].first.rows), stride) ||
                        (i + pyramid[index].first.rows > large_plane.rows) ||
                        (j + pyramid[index].first.cols > large_plane.cols)
                        )
                    {
                        break;
                    }
                }
                if (id != index)
                {
                    continue;
                }
                else
                {
                    pyramid[index].second = cv::Point(j, i);
                    pyramid[index].first.copyTo(large_plane(cv::Rect(pyramid[index].second.x,
                                                                     pyramid[index].second.y,
                                                                     pyramid[index].first.cols,
                                                                     pyramid[index].first.rows)));
                    is_copied = true;
                    break;
                }
            }
            if (is_copied)
            {
                break;
            }
        }
        ++index;
    }
    cv::resize(large_plane, large_plane, cv::Size(alex_size, alex_size));
    return large_plane;
}

void FeatureExtractor::modify_proto_file(const std::string &in,
                                         const std::string &out,
                                         const int rows, const int cols)
{
    std::ifstream protoIn(in, std::ios::in);
    std::ofstream protoOut(out, std::ios::out);
    
    auto index = 0;
    for (std::string line; std::getline(protoIn, line); ++index)
    {
        if (index == 5)
        {
            protoOut << "dim: " << rows << '\n';
        }
        else if (index == 6)
        {
            protoOut << "dim: " << cols << '\n';
        }
        else
        {
            protoOut << line << '\n';
        }
    }
    protoOut.close();
    protoIn.close();
}

FeatureExtractor::FeatureExtractor(const std::string &proto,
                          const std::string &caffe_model)
: _mean({red_channel_mean, green_channel_mean, blue_channel_mean}),
proto(proto), caffe_model(caffe_model)
{
    net.reset(new caffe::Net<double>(proto, caffe::Phase::TEST));
    net->CopyTrainedLayersFrom(caffe_model);
}

std::vector<double> FeatureExtractor::extract(const cv::Mat &image)
{
    assert(!image.empty());
    
//    cv::Mat processed_image =
//    pre_processing(image, {red_channel_mean,
//        green_channel_mean, blue_channel_mean});
    
    cv::Mat processed_image;
    cv::resize(image, processed_image, cv::Size(googlenet_size,
                                                googlenet_size));

    {
        std::vector<cv::Mat> channels;
        cv::split(processed_image, channels);
    
        std::vector<Eigen::MatrixXf> rgbImage;
        rgbImage.emplace_back(OpenCV2Eigen(channels[0]));
        rgbImage.emplace_back(OpenCV2Eigen(channels[1]));
        rgbImage.emplace_back(OpenCV2Eigen(channels[2]));
        
        rgbImage[0].array() -= red_channel_mean;
        rgbImage[1].array() -= green_channel_mean;
        rgbImage[2].array() -= blue_channel_mean;
        
        std::vector<std::vector<Eigen::MatrixXf>> rgbImages{std::move(rgbImage)};

        Eigen2Blob<double>(rgbImages, net);
        net->ForwardPrefilled();
    }
    
    caffe::Blob<double>* output_layer = net->output_blobs()[0];
    double *feature_vector = const_cast<double*>(output_layer->cpu_data());
    return std::vector<double>(feature_vector,
                               feature_vector + google_vector_length);
}

void split(std::vector<std::string>& elements, std::string str,
           const std::string& regex)
{
    elements.clear();
    std::regex re(regex);
    std::sregex_token_iterator first{str.begin(), str.end(), re, -1}, last;
    
    std::move(first, last, std::back_inserter(elements));
}

void split(std::vector<std::string>& elements, std::string str,
           char regex)
{
    split(elements, str, std::string{regex});
}