/*
 * Config.cpp
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#include "Config.h"

namespace myfusion {

Config::Config() {
	// TODO Auto-generated constructor stub

}

Config::~Config() {
	if(file_.isOpened())
		file_.release();
}

void Config::setParameterFile(const std::string& filename)
{
	if (config_ == nullptr)
		config_ = shared_ptr < Config > (new Config);
	config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
	if (config_->file_.isOpened() == false) {
		std::cerr << "parameter file " << filename << " does not exist."
				<< std::endl;
		config_->file_.release();
		return;
	}
}


shared_ptr<Config> Config::config_ = nullptr;

} /* namespace myfusion */
