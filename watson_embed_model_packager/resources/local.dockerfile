################################################################################
# This Dockerfile builds an image holding a single model a local path on disk
# and the tools needed to unpack it locally and then to optionally upload it to
# an s3 compatible storage
################################################################################

## build #######################################################################
#
# NOTE: We cannot guarantee that the automation platform using this build will
#   have buildkit available, so we cannot use COPY --chmod. Instead, we ensure
#   that permissions are set correctly in the build stage using RUN and then
#   copy the correctly-permissioned objects to the release phase.
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest as build

# Set up the additional package needs for build and run time
COPY copy_bin_with_libs.sh /copy_bin_with_libs.sh
ENV BIN_OUT_PATH=/bin_out
ENV LIB_OUT_PATH=/lib_out
RUN true && \
    microdnf update -y && \
    microdnf install -y which zip findutils openssl util-linux jq && \
    /copy_bin_with_libs.sh openssl && \
    /copy_bin_with_libs.sh curl && \
    /copy_bin_with_libs.sh sed && \
    /copy_bin_with_libs.sh unzip && \
    /copy_bin_with_libs.sh find && \
    /copy_bin_with_libs.sh rev && \
    /copy_bin_with_libs.sh jq && \
    chmod ugo+rx /bin_out/* && \
    chmod ugo+r /lib_out/* && \
    true

# Build-time arguments that won't go into the release
# e.g. sample_models/doc_conversion
ARG MODEL_PATH
ARG MODEL_DEST
COPY ${MODEL_PATH} ${MODEL_DEST}

# Zip up the model directory
RUN true && \
    cd / && \
    mv $(dirname ${MODEL_DEST})/* . && \
    cd $(basename ${MODEL_DEST}) && \
    zip -r /model.zip * && \
    chmod 444 /model.zip && \
    true

# Copy the build script and ensure that it's permissions are set correctly
COPY unpack_model.sh /unpack_model.sh
RUN chmod 555 /unpack_model.sh

## release #####################################################################
FROM registry.access.redhat.com/ubi9/ubi-micro:latest as release
ARG MODEL_NAME
ENV MODEL_ROOT_DIR="/app/models"
ENV MODEL_NAME=$MODEL_NAME

# Copy over the binaries needed to unpack
COPY --from=build /bin_out/* /usr/bin/
COPY --from=build /lib_out/* /lib64/

# Copy over CA certificate bundle for `curl` to work
COPY --from=build /etc/pki/tls/certs/ca-bundle.crt /etc/pki/tls/certs/ca-bundle.crt

# Copy over the model and the unpacking script
WORKDIR /app
RUN chmod -R ugo+rw /app/
COPY --from=build /model.zip /app/model.zip
COPY --from=build /unpack_model.sh /app/unpack_model.sh
USER 1001:0
ENTRYPOINT /app/unpack_model.sh