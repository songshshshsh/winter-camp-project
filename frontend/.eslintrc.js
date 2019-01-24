module.exports = {
  root: true,
  env: {
    node: true,
  },
  extends: [
    'plugin:vue/essential',
    '@vue/airbnb',
    '@vue/typescript',
  ],
  rules: {
    'class-methods-use-this': 'off',
    'object-curly-newline': 'off',
    'import/no-dynamic-require': 'off',
    'no-param-reassign': 'off',
    'global-require': 'off',
    'max-len': 'off',
    'no-nested-ternary': 'off',
    camelcase: 'off',
    'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
  },
  parserOptions: {
    parser: 'typescript-eslint-parser',
  },
};
