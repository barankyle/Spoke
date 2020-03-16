# build stage
FROM node:10 as build-stage

# RUN apk add -U --no-cache git
COPY . /spoke
WORKDIR /spoke
RUN yarn install --frozen-lockfile

RUN  yarn build
RUN mkdir -p dist/pages
# RUN mv dist/*.html dist/pages


# production stage
FROM nginx:stable-alpine as production-stage
COPY --from=build-stage /spoke/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
