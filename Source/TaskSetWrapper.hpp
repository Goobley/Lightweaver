#ifndef CMO_LW_TASK_SET_WRAPPER_HPP
#define CMO_LW_TASK_SET_WRAPPER_HPP
#include "TaskScheduler.h"
#include <cassert>

typedef void(*TaskSetFn)(void* userdata, enki::TaskScheduler* s,
                         enki::TaskSetPartition, uint32_t threadId);

struct LwTaskSet : public enki::ITaskSet
{
    LwTaskSet() = default;
    LwTaskSet(void* userdata_, enki::TaskScheduler* sched_,
              uint32_t setSize_, uint32_t minRange_, TaskSetFn fn_)
        : ITaskSet(setSize_, minRange_),
          taskFn(fn_),
          sched(sched_),
          userdata(userdata_)
    {}

    void ExecuteRange(enki::TaskSetPartition p, uint32_t threadId) override
    {
        if (!userdata)
            assert(false);
        taskFn(userdata, sched, p, threadId);
    }

    TaskSetFn taskFn;
    enki::TaskScheduler* sched;
    void* userdata;
};

#else
#endif