# Compile narrowly patched copies, leaving the pinned submodule untouched.
# Fail at configure time if an ncnn update changes any expected location.
function(rvf_patch_ncnn_source name)
    set(rvf_source "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/ncnn/src/${name}.cpp")
    file(READ "${rvf_source}" rvf_code)
    math(EXPR rvf_odd "${ARGC} % 2")
    if(ARGC LESS 3 OR NOT rvf_odd EQUAL 1)
        message(FATAL_ERROR "ncnn compatibility edits must be before/after pairs")
    endif()
    math(EXPR rvf_last "${ARGC} - 1")
    foreach(rvf_index RANGE 1 ${rvf_last} 2)
        math(EXPR rvf_next "${rvf_index} + 1")
        set(before "${ARGV${rvf_index}}")
        set(after "${ARGV${rvf_next}}")
        string(FIND "${rvf_code}" "${before}" rvf_position)
        if(rvf_position EQUAL -1)
            message(FATAL_ERROR "Pinned ncnn ${name} workaround no longer matches; review ncnn_compat.cmake")
        endif()
        string(LENGTH "${before}" rvf_length)
        math(EXPR rvf_end "${rvf_position} + ${rvf_length}")
        string(SUBSTRING "${rvf_code}" ${rvf_end} -1 rvf_suffix)
        string(FIND "${rvf_suffix}" "${before}" rvf_duplicate)
        if(NOT rvf_duplicate EQUAL -1)
            message(FATAL_ERROR "Pinned ncnn ${name} workaround is ambiguous")
        endif()
        string(REPLACE "${before}" "${after}" rvf_code "${rvf_code}")
    endforeach()
    set(rvf_patched "${CMAKE_CURRENT_BINARY_DIR}/compat/ncnn_${name}.cpp")
    file(GENERATE OUTPUT "${rvf_patched}" CONTENT "${rvf_code}")
    set_source_files_properties("${rvf_patched}" TARGET_DIRECTORY ncnn PROPERTIES GENERATED TRUE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${rvf_source}")
    set_source_files_properties("${rvf_source}" TARGET_DIRECTORY ncnn PROPERTIES HEADER_FILE_ONLY TRUE)
    target_sources(ncnn PRIVATE "${rvf_patched}")
endfunction()

# RX 550 advertises subgroup-size control but requiredSubgroupSizeStages == 0.
# Requesting a subgroup size for compute violates VUID ...-pNext-02755.
rvf_patch_ncnn_source(gpu
    "if (info.support_subgroup_size_control())"
    "if (info.support_subgroup_size_control()\n        && (info.querySubgroupSizeControlProperties().requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))")

# MemoryData constants can be cloned before an in-place operation. Their weight
# buffers must be legal copy sources (VUID-vkCmdCopyBuffer-srcBuffer-00118).
rvf_patch_ncnn_source(allocator
    "create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)"
    "create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)")

# Recycled blob suballocations lose the previous access state. Treat that state
# as unknown, not unwritten. Also synchronize copy destinations before writing.
# Both immediate (push-descriptor) and delayed recording paths are handled.
rvf_patch_ncnn_source(command
    [=[void VkCompute::barrier_readwrite(const VkMat& binding)
{]=]
    [=[void VkCompute::barrier_readwrite(const VkMat& binding)
{
    if (binding.data->stage_flags == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)
    {
        binding.data->access_flags = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        binding.data->stage_flags = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    }]=]
    [=[    //     NCNN_LOGE("record_clone buffer to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;]=]
    [=[    //     NCNN_LOGE("record_clone buffer to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // A newly returned pool span can alias an earlier GPU read or write.
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = dst.buffer();
        barriers[0].offset = dst.data->offset;
        barriers[0].size = dst.data->capacity;
        const VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        const VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.command_buffer = d->compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }
    }]=])
