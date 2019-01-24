const proxy = require('./proxyTable');

module.exports = {
  devServer: { proxy },
  runtimeCompiler: true,
  assetsDir: 'frontend_static',
};
