# The operator UI: built once, served as static files by nginx.
#
# The API base is baked in at build time because Vite inlines `import.meta.env` -- there is no
# runtime config to read. nginx proxies /api to the service instead, so the bundle ships with a
# same-origin base and CORS is not involved in production at all.

FROM node:24-alpine AS build

WORKDIR /build
COPY app/package.json app/package-lock.json ./
RUN npm ci

COPY app/ ./
ENV VITE_API=/api
RUN npm run build

FROM nginx:1.29-alpine AS serve

COPY --from=build /build/dist /usr/share/nginx/html
COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
