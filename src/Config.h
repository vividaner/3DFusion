/*
 * Config.h
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include "common_include.h"

namespace myfusion {

class Config {
public:
	Config();
	virtual ~Config();

	static void setParameterFile(const std::string& filename);
	template<typename T>
	static T get(const std::string& key)
	{
		return T(Config::config_->file_[key]);
	}

private:
	static std::shared_ptr<Config> config_;
	cv::FileStorage file_;

};

} /* namespace myfusion */

#endif /* CONFIG_H_ */
