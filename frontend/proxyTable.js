/**
 * Replace it by your server addr
 */
const target = 'http://34.80.92.40:8887';

module.exports = {
  '/api': {
    target,
    changeOrigin: true,
    pathRewrite: {
      '^/api': '/api',
    },
  },
};
