module.exports = {
  presets: ['module:metro-react-native-babel-preset', '@babel/preset-react'],
  plugins: [
    [
      'module:react-native-dotenv',
      {
        envName: 'APP_ENV',
        moduleName: '@env',
        path: '.env',
      },
    ],
  ],
};
