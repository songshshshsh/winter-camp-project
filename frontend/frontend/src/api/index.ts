// eslint-disable-next-line
/// <reference path="./index.d.ts" />

import axios, { AxiosPromise } from 'axios';
import qs from 'qs';

const request = axios.create({
  baseURL: '/api',
});

/**
 * [请求拦截器]
 */
request.interceptors.request.use(config => config, err => Promise.reject(err));

/**
 * [回复拦截器]
 * 用于错误处理，只要请求出错，则利用 element-ui 提供的弹窗在浏览器
 * 中打出错误内容。
 */
request.interceptors.response.use(res => res, err => Promise.reject(err));

const api = {
  getClass: (param: TextArgs): AxiosPromise<TextRes> => (
    request.get(`/classification?text=${param.text}`)
  ),
  getToken: (param: RunArgs): AxiosPromise<RunRes> => (
    request.get(`/predict?text=${param.text}&label0=${param.label0}&label1=${param.label1}`)
  ),
};

export default api;
