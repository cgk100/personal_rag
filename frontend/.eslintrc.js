module.exports = {
  root: true,
  env: {
    node: true,
    'vue/setup-compiler-macros': true
  },
  extends: [
    'plugin:vue/vue3-essential',
    'eslint:recommended'
  ],
  parserOptions: {
    parser: '@babel/eslint-parser',
    requireConfigFile: false,
    ecmaVersion: 2020
  },
  rules: {
    'no-undef': 'off',  // 临时禁用 no-undef 规则
    'vue/no-setup-props-destructure': 'off',
    'vue/multi-word-component-names': 'off'
  }
} 