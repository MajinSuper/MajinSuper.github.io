# 确保脚本抛出遇到的错误
set -e

## 安装依赖
# npm install

# 运行构建命令，生成静态文件
npm run docs:build

# 进入生成的文件夹
cd docs/.vuepress/dist

# 推送构建结果到gh-pages分支
git init
# 邮箱
git config user.email "1533363937@qq.com"
# 用户名
git config --global user.name "MajinSuper"
# 添加文件并提交
git add -A
git commit -m 'Automatic build'

# 不确定 git init 后，当前分支是master 还是 main
# 这里再次强制创建切换到 master 分支。
# 可以指定其他的分支名，能和 push 命令里的分支名对应即可。
git checkout -B master
# 强制推送到gh-pages分支。
git push -f git@github.com:MajinSuper/MajinSuper.github.io.git master:gh-pages

cd -