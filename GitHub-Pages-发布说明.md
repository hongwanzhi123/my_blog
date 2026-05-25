# GitHub Pages 发布说明

这个目录已经整理成适合发布个人博客的静态站点结构。

## 当前入口

- 首页：`index.html`
- 样式：`styles.css`
- 博客原始整理文档：`assets/个人博客/`

## 最推荐的发布方式

把整个 `算法设计笔记` 文件夹上传到一个新的 GitHub 仓库，然后开启 GitHub Pages。

## 操作步骤

1. 在 GitHub 新建一个公开仓库。
2. 仓库名建议使用英文，例如：`algorithm-notes-portfolio`
3. 把当前整个文件夹内容上传到仓库根目录。
4. 进入 GitHub 仓库页面。
5. 打开 `Settings`。
6. 打开 `Pages`。
7. 在 `Build and deployment` 里选择：
   - `Source`: `Deploy from a branch`
   - `Branch`: `main`
   - `Folder`: `/ (root)`
8. 保存后等待 GitHub 完成部署。

## 部署后的网址格式

如果你的 GitHub 用户名是 `yourname`，仓库名是 `algorithm-notes-portfolio`，那么网址通常会是：

`https://yourname.github.io/algorithm-notes-portfolio/`

## 简历里建议填写的写法

个人博客：

`https://yourname.github.io/algorithm-notes-portfolio/`

## 说明

- 这个站点是纯静态页面，不需要安装依赖。
- `.nojekyll` 文件已经加好，可以避免 GitHub Pages 对部分静态资源做不必要处理。
- 如果后续你想把它升级成更完整的博客系统，也可以继续迁移到 Vercel 或独立域名。
