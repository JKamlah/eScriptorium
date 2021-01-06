const path = require('path');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  entry: {
    vendor: './src/vendor.js',
    main: './src/main.js',
    editor: './src/editor/main.js',
  },

  output: {
    filename: '[name].js',
    path: path.resolve(__dirname, './dist/'),
    publicPath: '',
  },

  plugins: [
    new MiniCssExtractPlugin(),
  ],

  module: {
    rules: [
      { test: /\.css$/,
        use: [MiniCssExtractPlugin.loader, 'css-loader'],
      },
      { test: /\.(png|jpe?g|gif|woff|woff2|eot|ttf|otf|svg)$/i,
        use: ['file-loader'],
      }
    ]
  }
};
