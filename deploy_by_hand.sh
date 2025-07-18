#!/usr/bin/env sh
 
# 忽略错误
set -e  #有错误抛出错误

# 构建
pnpm run docs:build  #然后执行打包命令
 
# 进入待发布的目录
cd docs/.vuepress/dist  #进到dist目录
 
git init  #执行这些git命令
git add -A
git commit -m 'deploy'
 
# 上传仓库，这里改成你自己的仓库
git push -f git@github.com:MajinSuper/MajinSuper.github.io.git master:gh-pages
 
cd -
 
rm -rf docs/.vuepress/dist  #删除dist文件夹