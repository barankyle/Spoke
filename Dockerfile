# build stage
FROM node:10.16 as build-stage


# hubs modules
WORKDIR /spoke
COPY package*.json /spoke/
COPY yarn.lock /spoke/
RUN yarn install --frozen-lockfile

COPY . /spoke/
WORKDIR /spoke

ARG NODE_TLS_REJECT_UNAUTHORIZED=0
ARG ROUTER_BASE_PATH=/spoke
ARG BASE_ASSETS_PATH=https://hubs.local:9090/ 
ARG HUBS_SERVER=hubs.local:4000
ARG RETICULUM_SERVER=hubs.local:4000

RUN  yarn build
RUN mkdir -p dist/pages
# RUN mv dist/*.html dist/pages


# production stage
FROM nginx:stable-alpine as production-stage
COPY --from=build-stage /spoke/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
